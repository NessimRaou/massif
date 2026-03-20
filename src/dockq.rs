use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::time::Instant;

use indicatif::ParallelProgressIterator;
use nalgebra::{Matrix3, Vector3};
use pdbtbx::{Element, PDB, ReadOptions, StrictnessLevel};
use rayon::prelude::*;

use crate::progress::default_progress_style;

const BACKBONE_ATOMS: [&str; 16] = [
    "CA", "C", "N", "O", "P", "OP1", "OP2", "O2'", "O3'", "O4'", "O5'", "C1'", "C2'", "C3'",
    "C4'", "C5'",
];
const CLASH_CUTOFF: f64 = 2.0;
const IRMSD_SCALE: f64 = 1.5;
const LRMSD_SCALE: f64 = 8.5;

#[derive(Clone, Debug)]
pub struct DockQConfig {
    pub contact_cutoff: f64,
    pub interface_cutoff: f64,
}

impl Default for DockQConfig {
    fn default() -> Self {
        Self {
            contact_cutoff: 5.0,
            interface_cutoff: 10.0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DockQPartners {
    pub receptor: String,
    pub ligand: String,
}

#[derive(Clone, Debug)]
pub struct DockQResult {
    pub dockq: f64,
    pub fnat: f64,
    pub irmsd: f64,
    pub lrmsd: f64,
    pub native_contacts: usize,
    pub model_contacts: usize,
    pub shared_contacts: usize,
    pub nonnative_contacts: usize,
    pub clashes: usize,
}

#[derive(Clone, Debug)]
pub struct DockQBatchResult {
    pub model: String,
    pub result: Result<DockQResult, DockQError>,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResidueKey {
    pub chain_id: String,
    pub residue_number: isize,
    pub insertion_code: Option<String>,
    pub residue_name: String,
}

#[derive(Clone, Debug)]
pub enum DockQError {
    InvalidPartners(String),
    MissingChain {
        structure_role: &'static str,
        chain_id: String,
    },
    StructureRead {
        path: String,
        message: String,
    },
    EmptyPartner {
        structure_role: &'static str,
        partner_role: &'static str,
        chain_group: String,
    },
    NoAlignedResidues {
        chain_id: String,
    },
    NoNativeContacts,
    EmptyAtomSelection {
        selection: &'static str,
    },
    AtomCountMismatch {
        selection: &'static str,
        native_atoms: usize,
        model_atoms: usize,
    },
}

impl fmt::Display for DockQError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DockQError::InvalidPartners(message) => write!(f, "{message}"),
            DockQError::MissingChain {
                structure_role,
                chain_id,
            } => write!(f, "{structure_role} is missing required chain '{chain_id}'"),
            DockQError::StructureRead { path, message } => {
                write!(f, "failed to read structure '{path}': {message}")
            }
            DockQError::EmptyPartner {
                structure_role,
                partner_role,
                chain_group,
            } => write!(
                f,
                "{structure_role} {partner_role} group '{chain_group}' did not yield any residues"
            ),
            DockQError::NoAlignedResidues { chain_id } => write!(
                f,
                "no identical aligned residues were found for chain '{chain_id}' between model and reference"
            ),
            DockQError::NoNativeContacts => {
                write!(f, "reference complex does not contain any native receptor-ligand contacts")
            }
            DockQError::EmptyAtomSelection { selection } => {
                write!(f, "no atoms were available for DockQ {selection}")
            }
            DockQError::AtomCountMismatch {
                selection,
                native_atoms,
                model_atoms,
            } => write!(
                f,
                "DockQ {selection} atom selection mismatch (reference: {native_atoms}, model: {model_atoms})"
            ),
        }
    }
}

impl Error for DockQError {}

#[derive(Clone, Debug)]
struct ResidueRecord {
    key: ResidueKey,
    sequence_code: char,
    all_atoms: Vec<Vector3<f64>>,
    atoms_by_name: HashMap<String, Vector3<f64>>,
}

#[derive(Clone, Debug)]
struct StructureView {
    chains: HashMap<String, Vec<ResidueRecord>>,
}

#[derive(Clone, Debug)]
struct AlignedPartner {
    model: Vec<ResidueRecord>,
    native: Vec<ResidueRecord>,
}

#[derive(Clone, Debug)]
struct ResidueDistanceMatrix {
    n_rows: usize,
    n_cols: usize,
    min_sq_distances: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ContactStats {
    native_contacts: usize,
    model_contacts: usize,
    shared_contacts: usize,
    nonnative_contacts: usize,
}

impl ResidueDistanceMatrix {
    fn new(n_rows: usize, n_cols: usize, min_sq_distances: Vec<f64>) -> Self {
        Self {
            n_rows,
            n_cols,
            min_sq_distances,
        }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self.min_sq_distances[row * self.n_cols + col]
    }

    fn count_below(&self, threshold_sq: f64) -> usize {
        self.min_sq_distances
            .iter()
            .filter(|distance| **distance < threshold_sq)
            .count()
    }

    fn compare_contacts(&self, native: &Self, threshold_sq: f64) -> ContactStats {
        let mut stats = ContactStats::default();
        for (model_distance, native_distance) in self
            .min_sq_distances
            .iter()
            .zip(native.min_sq_distances.iter())
        {
            let model_contact = *model_distance < threshold_sq;
            let native_contact = *native_distance < threshold_sq;

            if native_contact {
                stats.native_contacts += 1;
            }
            if model_contact {
                stats.model_contacts += 1;
            }
            if model_contact && native_contact {
                stats.shared_contacts += 1;
            }
            if model_contact && !native_contact {
                stats.nonnative_contacts += 1;
            }
        }
        stats
    }

    fn interface_indices(&self, threshold_sq: f64) -> (HashSet<usize>, HashSet<usize>) {
        let mut receptor_residues = HashSet::new();
        let mut ligand_residues = HashSet::new();

        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                if self.get(row, col) < threshold_sq {
                    receptor_residues.insert(row);
                    ligand_residues.insert(col);
                }
            }
        }

        (receptor_residues, ligand_residues)
    }
}

pub fn compute_dockq(
    reference: &PDB,
    model: &PDB,
    partners: &DockQPartners,
    config: &DockQConfig,
) -> Result<DockQResult, DockQError> {
    validate_partners(partners)?;

    let native_view = StructureView::from_pdb(reference);
    let model_view = StructureView::from_pdb(model);

    let native_receptor = native_view.select_partner(&partners.receptor, "reference", "receptor")?;
    let model_receptor = model_view.select_partner(&partners.receptor, "model", "receptor")?;
    let native_ligand = native_view.select_partner(&partners.ligand, "reference", "ligand")?;
    let model_ligand = model_view.select_partner(&partners.ligand, "model", "ligand")?;

    let aligned_receptor = align_partner_group(&model_receptor, &native_receptor)?;
    let aligned_ligand = align_partner_group(&model_ligand, &native_ligand)?;

    let native_matrix =
        compute_residue_distance_matrix(&aligned_receptor.native, &aligned_ligand.native);
    let model_matrix =
        compute_residue_distance_matrix(&aligned_receptor.model, &aligned_ligand.model);

    let contact_threshold_sq = config.contact_cutoff * config.contact_cutoff;
    let interface_threshold_sq = config.interface_cutoff * config.interface_cutoff;

    let contact_stats = model_matrix.compare_contacts(&native_matrix, contact_threshold_sq);
    if contact_stats.native_contacts == 0 {
        return Err(DockQError::NoNativeContacts);
    }

    let (receptor_interface, ligand_interface) =
        native_matrix.interface_indices(interface_threshold_sq);

    let irmsd = compute_irmsd(
        &aligned_receptor,
        &aligned_ligand,
        &receptor_interface,
        &ligand_interface,
    )?;
    let lrmsd = compute_lrmsd(&aligned_receptor, &aligned_ligand)?;
    let fnat = contact_stats.shared_contacts as f64 / contact_stats.native_contacts as f64;
    let clashes = model_matrix.count_below(CLASH_CUTOFF * CLASH_CUTOFF);

    Ok(DockQResult {
        dockq: dockq_formula(fnat, irmsd, lrmsd),
        fnat,
        irmsd,
        lrmsd,
        native_contacts: contact_stats.native_contacts,
        model_contacts: contact_stats.model_contacts,
        shared_contacts: contact_stats.shared_contacts,
        nonnative_contacts: contact_stats.nonnative_contacts,
        clashes,
    })
}

pub fn compute_dockq_from_files(
    reference_path: &str,
    model_path: &str,
    partners: &DockQPartners,
    config: &DockQConfig,
) -> Result<DockQResult, DockQError> {
    let reference = read_structure(reference_path)?;
    let model = read_structure(model_path)?;
    compute_dockq(&reference, &model, partners, config)
}

pub fn all_dockq(
    reference_path: &str,
    pdb_file_names: &[String],
    input_dir: &str,
    partners: &DockQPartners,
    config: &DockQConfig,
) -> Result<Vec<DockQBatchResult>, DockQError> {
    println!(
        "Computing DockQ against reference {} on {} structures...",
        reference_path,
        pdb_file_names.len()
    );
    let start = Instant::now();
    let style = default_progress_style();
    let reference = read_structure(reference_path)?;

    let results = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|model_name| {
            let model_path = format!("{input_dir}/{model_name}");
            let result = match read_structure(&model_path) {
                Ok(model) => compute_dockq(&reference, &model, partners, config),
                Err(err) => Err(err),
            };
            DockQBatchResult {
                model: model_name.clone(),
                result,
            }
        })
        .collect();

    println!("Took {:?}\n", start.elapsed());
    Ok(results)
}

impl StructureView {
    fn from_pdb(pdb: &PDB) -> Self {
        let mut chains = HashMap::new();

        for chain in pdb.chains() {
            let mut residues = Vec::new();

            for residue in chain.residues() {
                let residue_name = residue.name().unwrap_or("UNK").trim().to_string();
                let mut atoms_by_name = HashMap::new();
                let mut all_atoms = Vec::new();

                for atom in residue.atoms() {
                    if atom.element() == Some(&Element::H) {
                        continue;
                    }

                    let atom_name = atom.name().trim().to_string();
                    if atoms_by_name.contains_key(&atom_name) {
                        continue;
                    }

                    let (x, y, z) = atom.pos();
                    let position = Vector3::new(x, y, z);
                    atoms_by_name.insert(atom_name, position);
                    all_atoms.push(position);
                }

                if all_atoms.is_empty() {
                    continue;
                }

                let key = ResidueKey {
                    chain_id: chain.id().to_string(),
                    residue_number: residue.serial_number(),
                    insertion_code: residue.insertion_code().map(|code| code.to_string()),
                    residue_name: residue_name.clone(),
                };

                residues.push(ResidueRecord {
                    sequence_code: residue_to_code(&residue_name),
                    key,
                    all_atoms,
                    atoms_by_name,
                });
            }

            chains.insert(chain.id().to_string(), residues);
        }

        Self { chains }
    }

    fn select_partner(
        &self,
        chain_group: &str,
        structure_role: &'static str,
        partner_role: &'static str,
    ) -> Result<Vec<Vec<ResidueRecord>>, DockQError> {
        let mut selected = Vec::new();
        let chain_ids = parse_chain_group(chain_group);

        for chain_id in chain_ids {
            let residues = self
                .chains
                .get(&chain_id)
                .ok_or_else(|| DockQError::MissingChain {
                    structure_role,
                    chain_id: chain_id.clone(),
                })?;
            if !residues.is_empty() {
                selected.push(residues.clone());
            }
        }

        if selected.is_empty() {
            return Err(DockQError::EmptyPartner {
                structure_role,
                partner_role,
                chain_group: chain_group.to_string(),
            });
        }

        Ok(selected)
    }
}

fn validate_partners(partners: &DockQPartners) -> Result<(), DockQError> {
    let receptor_chains = parse_chain_group(&partners.receptor);
    let ligand_chains = parse_chain_group(&partners.ligand);

    if receptor_chains.is_empty() || ligand_chains.is_empty() {
        return Err(DockQError::InvalidPartners(
            "receptor and ligand chain groups must each contain at least one chain".to_string(),
        ));
    }

    let receptor_set: HashSet<String> = receptor_chains.into_iter().collect();
    let ligand_set: HashSet<String> = ligand_chains.into_iter().collect();
    if receptor_set.intersection(&ligand_set).next().is_some() {
        return Err(DockQError::InvalidPartners(
            "receptor and ligand chain groups must be disjoint".to_string(),
        ));
    }

    Ok(())
}

fn read_structure(path: &str) -> Result<PDB, DockQError> {
    let mut pdb = ReadOptions::default()
        .set_level(StrictnessLevel::Loose)
        .read(path)
        .map_err(|errors| DockQError::StructureRead {
            path: path.to_string(),
            message: errors
                .iter()
                .map(|error| error.to_string())
                .collect::<Vec<String>>()
                .join("; "),
        })?
        .0;
    pdb.remove_atoms_by(|atom| atom.element() == Some(&Element::H));
    pdb.full_sort();
    Ok(pdb)
}

fn parse_chain_group(chain_group: &str) -> Vec<String> {
    if chain_group.chars().any(|c| c == ',' || c.is_whitespace()) {
        chain_group
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|id| !id.is_empty())
            .map(|id| id.to_string())
            .collect()
    } else {
        chain_group.chars().map(|c| c.to_string()).collect()
    }
}

fn align_partner_group(
    model_chains: &[Vec<ResidueRecord>],
    native_chains: &[Vec<ResidueRecord>],
) -> Result<AlignedPartner, DockQError> {
    let mut model = Vec::new();
    let mut native = Vec::new();

    for (model_chain, native_chain) in model_chains.iter().zip(native_chains.iter()) {
        let aligned_pairs = align_chain_residues(model_chain, native_chain)?;
        for (model_residue, native_residue) in aligned_pairs {
            model.push(model_residue);
            native.push(native_residue);
        }
    }

    Ok(AlignedPartner { model, native })
}

fn align_chain_residues(
    model_residues: &[ResidueRecord],
    native_residues: &[ResidueRecord],
) -> Result<Vec<(ResidueRecord, ResidueRecord)>, DockQError> {
    let chain_id = native_residues
        .first()
        .map(|residue| residue.key.chain_id.clone())
        .or_else(|| model_residues.first().map(|residue| residue.key.chain_id.clone()))
        .unwrap_or_else(|| String::from("?"));

    let model_sequence: Vec<char> = model_residues.iter().map(|residue| residue.sequence_code).collect();
    let native_sequence: Vec<char> = native_residues
        .iter()
        .map(|residue| residue.sequence_code)
        .collect();
    let aligned_pairs = identical_alignment_pairs(&model_sequence, &native_sequence);

    if aligned_pairs.is_empty() {
        return Err(DockQError::NoAlignedResidues { chain_id });
    }

    Ok(aligned_pairs
        .into_iter()
        .map(|(model_index, native_index)| {
            (
                model_residues[model_index].clone(),
                native_residues[native_index].clone(),
            )
        })
        .collect())
}

fn identical_alignment_pairs(model: &[char], native: &[char]) -> Vec<(usize, usize)> {
    const MATCH_SCORE: i32 = 2;
    const MISMATCH_SCORE: i32 = -1;
    const GAP_SCORE: i32 = -2;

    let mut scores = vec![vec![0_i32; native.len() + 1]; model.len() + 1];

    for i in 1..=model.len() {
        scores[i][0] = scores[i - 1][0] + GAP_SCORE;
    }
    for j in 1..=native.len() {
        scores[0][j] = scores[0][j - 1] + GAP_SCORE;
    }

    for i in 1..=model.len() {
        for j in 1..=native.len() {
            let diagonal = scores[i - 1][j - 1]
                + if model[i - 1] == native[j - 1] {
                    MATCH_SCORE
                } else {
                    MISMATCH_SCORE
                };
            let up = scores[i - 1][j] + GAP_SCORE;
            let left = scores[i][j - 1] + GAP_SCORE;
            scores[i][j] = diagonal.max(up).max(left);
        }
    }

    let mut aligned_pairs = Vec::new();
    let mut i = model.len();
    let mut j = native.len();

    while i > 0 || j > 0 {
        let diagonal_score = if i > 0 && j > 0 {
            Some(
                scores[i - 1][j - 1]
                    + if model[i - 1] == native[j - 1] {
                        MATCH_SCORE
                    } else {
                        MISMATCH_SCORE
                    },
            )
        } else {
            None
        };
        let up_score = if i > 0 {
            Some(scores[i - 1][j] + GAP_SCORE)
        } else {
            None
        };

        if let Some(diagonal) = diagonal_score {
            if scores[i][j] == diagonal {
                i -= 1;
                j -= 1;
                if model[i] == native[j] {
                    aligned_pairs.push((i, j));
                }
                continue;
            }
        }

        if let Some(up) = up_score {
            if scores[i][j] == up {
                i -= 1;
                continue;
            }
        }

        if j > 0 {
            j -= 1;
        }
    }

    aligned_pairs.reverse();
    aligned_pairs
}

fn compute_residue_distance_matrix(
    receptor: &[ResidueRecord],
    ligand: &[ResidueRecord],
) -> ResidueDistanceMatrix {
    let mut distances = Vec::with_capacity(receptor.len() * ligand.len());

    for receptor_residue in receptor {
        for ligand_residue in ligand {
            distances.push(min_squared_distance(
                &receptor_residue.all_atoms,
                &ligand_residue.all_atoms,
            ));
        }
    }

    ResidueDistanceMatrix::new(receptor.len(), ligand.len(), distances)
}

fn min_squared_distance(first: &[Vector3<f64>], second: &[Vector3<f64>]) -> f64 {
    let mut min_distance = f64::INFINITY;

    for atom_a in first {
        for atom_b in second {
            let diff = atom_a - atom_b;
            let distance = diff.dot(&diff);
            if distance < min_distance {
                min_distance = distance;
            }
        }
    }

    min_distance
}

fn compute_irmsd(
    receptor: &AlignedPartner,
    ligand: &AlignedPartner,
    receptor_interface: &HashSet<usize>,
    ligand_interface: &HashSet<usize>,
) -> Result<f64, DockQError> {
    let (mut model_atoms, mut native_atoms) =
        collect_backbone_atom_pairs(receptor, Some(receptor_interface), "interface residues")?;
    let (ligand_model_atoms, ligand_native_atoms) =
        collect_backbone_atom_pairs(ligand, Some(ligand_interface), "interface residues")?;
    model_atoms.extend(ligand_model_atoms);
    native_atoms.extend(ligand_native_atoms);

    let (_, _, irmsd) = fit_and_rmsd(&native_atoms, &model_atoms, "interface superposition")?;
    Ok(irmsd)
}

fn compute_lrmsd(receptor: &AlignedPartner, ligand: &AlignedPartner) -> Result<f64, DockQError> {
    let (receptor_model_atoms, receptor_native_atoms) =
        collect_backbone_atom_pairs(receptor, None, "receptor superposition")?;
    let (ligand_model_atoms, ligand_native_atoms) =
        collect_backbone_atom_pairs(ligand, None, "ligand RMSD")?;

    let (rotation, translation, _) = fit_and_rmsd(
        &receptor_native_atoms,
        &receptor_model_atoms,
        "receptor superposition",
    )?;
    let transformed_ligand = transform_atoms(&ligand_model_atoms, rotation, translation);
    rmsd_without_superposition(&ligand_native_atoms, &transformed_ligand, "ligand RMSD")
}

fn collect_backbone_atom_pairs(
    partner: &AlignedPartner,
    residue_subset: Option<&HashSet<usize>>,
    selection: &'static str,
) -> Result<(Vec<Vector3<f64>>, Vec<Vector3<f64>>), DockQError> {
    let mut model_atoms = Vec::new();
    let mut native_atoms = Vec::new();

    for (index, (model_residue, native_residue)) in
        partner.model.iter().zip(partner.native.iter()).enumerate()
    {
        if residue_subset.is_some_and(|subset| !subset.contains(&index)) {
            continue;
        }

        for atom_name in BACKBONE_ATOMS {
            if let (Some(model_atom), Some(native_atom)) = (
                model_residue.atoms_by_name.get(atom_name),
                native_residue.atoms_by_name.get(atom_name),
            ) {
                model_atoms.push(*model_atom);
                native_atoms.push(*native_atom);
            }
        }
    }

    if model_atoms.is_empty() || native_atoms.is_empty() {
        return Err(DockQError::EmptyAtomSelection { selection });
    }
    if model_atoms.len() != native_atoms.len() {
        return Err(DockQError::AtomCountMismatch {
            selection,
            native_atoms: native_atoms.len(),
            model_atoms: model_atoms.len(),
        });
    }

    Ok((model_atoms, native_atoms))
}

fn fit_and_rmsd(
    native_atoms: &[Vector3<f64>],
    model_atoms: &[Vector3<f64>],
    selection: &'static str,
) -> Result<(Matrix3<f64>, Vector3<f64>, f64), DockQError> {
    if native_atoms.is_empty() || model_atoms.is_empty() {
        return Err(DockQError::EmptyAtomSelection { selection });
    }
    if native_atoms.len() != model_atoms.len() {
        return Err(DockQError::AtomCountMismatch {
            selection,
            native_atoms: native_atoms.len(),
            model_atoms: model_atoms.len(),
        });
    }

    let native_centroid = centroid(native_atoms);
    let model_centroid = centroid(model_atoms);
    let native_centered: Vec<Vector3<f64>> =
        native_atoms.iter().map(|atom| atom - native_centroid).collect();
    let model_centered: Vec<Vector3<f64>> =
        model_atoms.iter().map(|atom| atom - model_centroid).collect();

    let rotation = kabsch(&native_centered, &model_centered);
    let translation = native_centroid - rotation * model_centroid;
    let transformed_model = transform_atoms(model_atoms, rotation, translation);
    let rmsd = rmsd_without_superposition(native_atoms, &transformed_model, selection)?;

    Ok((rotation, translation, rmsd))
}

fn transform_atoms(
    atoms: &[Vector3<f64>],
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
) -> Vec<Vector3<f64>> {
    atoms.iter().map(|atom| rotation * atom + translation).collect()
}

fn rmsd_without_superposition(
    native_atoms: &[Vector3<f64>],
    model_atoms: &[Vector3<f64>],
    selection: &'static str,
) -> Result<f64, DockQError> {
    if native_atoms.is_empty() || model_atoms.is_empty() {
        return Err(DockQError::EmptyAtomSelection { selection });
    }
    if native_atoms.len() != model_atoms.len() {
        return Err(DockQError::AtomCountMismatch {
            selection,
            native_atoms: native_atoms.len(),
            model_atoms: model_atoms.len(),
        });
    }

    let squared_sum: f64 = native_atoms
        .iter()
        .zip(model_atoms.iter())
        .map(|(native_atom, model_atom)| {
            let diff = native_atom - model_atom;
            diff.dot(&diff)
        })
        .sum();

    Ok((squared_sum / native_atoms.len() as f64).sqrt())
}

fn centroid(atoms: &[Vector3<f64>]) -> Vector3<f64> {
    let sum = atoms.iter().fold(Vector3::zeros(), |acc, atom| acc + atom);
    sum / atoms.len() as f64
}

fn kabsch(reference: &[Vector3<f64>], sample: &[Vector3<f64>]) -> Matrix3<f64> {
    let mut covariance = Matrix3::zeros();
    for (reference_atom, sample_atom) in reference.iter().zip(sample.iter()) {
        covariance += reference_atom * sample_atom.transpose();
    }

    let svd = covariance.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let reflection = if (u * v_t).determinant() < 0.0 {
        Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0))
    } else {
        Matrix3::identity()
    };

    u * reflection * v_t
}

fn dockq_formula(fnat: f64, irmsd: f64, lrmsd: f64) -> f64 {
    (fnat + 1.0 / (1.0 + (irmsd / IRMSD_SCALE).powi(2))
        + 1.0 / (1.0 + (lrmsd / LRMSD_SCALE).powi(2)))
        / 3.0
}

fn residue_to_code(residue_name: &str) -> char {
    match residue_name {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" | "MSE" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        "A" | "DA" => 'A',
        "C" | "DC" => 'C',
        "G" | "DG" => 'G',
        "U" | "DT" | "DU" | "T" => 'T',
        _ => 'X',
    }
}

#[cfg(test)]
mod tests {
    use super::{dockq_formula, identical_alignment_pairs, ResidueDistanceMatrix};

    #[test]
    fn dockq_formula_matches_reference_definition() {
        let score = dockq_formula(1.0, 0.0, 0.0);
        assert!((score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn alignment_only_keeps_identical_matches() {
        let model = vec!['A', 'B', 'C', 'D'];
        let native = vec!['A', 'X', 'C', 'D'];
        let aligned = identical_alignment_pairs(&model, &native);
        assert_eq!(aligned, vec![(0, 0), (2, 2), (3, 3)]);
    }

    #[test]
    fn residue_distance_matrix_counts_contacts_correctly() {
        let model = ResidueDistanceMatrix::new(2, 2, vec![4.0, 36.0, 16.0, 1.0]);
        let native = ResidueDistanceMatrix::new(2, 2, vec![4.0, 81.0, 49.0, 1.0]);
        let stats = model.compare_contacts(&native, 25.0);
        assert_eq!(stats.native_contacts, 2);
        assert_eq!(stats.model_contacts, 3);
        assert_eq!(stats.shared_contacts, 2);
        assert_eq!(stats.nonnative_contacts, 1);
    }
}
