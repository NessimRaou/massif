use std::{fs, io, process, time::Instant};

use indicatif::{ParallelProgressIterator, ProgressIterator};
use nalgebra::{Matrix3, Vector3};
use pdbtbx::{save, Element, TransformationMatrix, PDB};
use rayon::prelude::*;

use crate::progress::default_progress_style;

/// Align two pdb structures using a ref chain as an anchor to align both on
fn align_structures_ref(pdb1: &PDB, pdb2: &mut PDB, ref_chain: &str, applying: &str) {
    let (rotation, translation) = compute_alignment_transform(pdb1, pdb2, ref_chain);
    match applying {
        "per_atom" => apply_transform(pdb2, rotation, translation),
        "on_pdb" | "on_pdb_par" => apply_transform_matrix(pdb2, rotation, translation),
        _ => println!("No transformation method called \"{applying}\""),
    }
}

/// get rotation and transformation matrix for optimal structure alignment using kabsch algorithm
pub fn try_compute_alignment_transform(
    pdb1: &PDB,
    pdb2: &PDB,
    ref_chain: &str,
) -> Result<(Matrix3<f64>, Vector3<f64>), String> {
    let atoms1 = try_collect_atom_positions_ref(pdb1, ref_chain)?;
    let atoms2 = try_collect_atom_positions_ref(pdb2, ref_chain)?;

    let centroid1 = compute_centroid(&atoms1);
    let centroid2 = compute_centroid(&atoms2);

    let atoms1_centered: Vec<Vector3<f64>> = atoms1.iter().map(|p| p - centroid1).collect();
    let atoms2_centered: Vec<Vector3<f64>> = atoms2.iter().map(|p| p - centroid2).collect();

    let rotation_matrix = kabsch(&atoms1_centered, &atoms2_centered);
    let translation_vector = centroid1 - (rotation_matrix * centroid2);
    Ok((rotation_matrix, translation_vector))
}

pub fn compute_alignment_transform(
    pdb1: &PDB,
    pdb2: &PDB,
    ref_chain: &str,
) -> (Matrix3<f64>, Vector3<f64>) {
    match try_compute_alignment_transform(pdb1, pdb2, ref_chain) {
        Ok(transform) => transform,
        Err(message) => {
            println!("{message}");
            process::exit(1);
        }
    }
}

/// get centroid from a structure
pub fn compute_centroid(atoms: &[Vector3<f64>]) -> Vector3<f64> {
    let sum = atoms.iter().fold(Vector3::zeros(), |acc, &p| acc + p);
    sum / (atoms.len() as f64)
}

/// calculation of the optimal rotation matrix that minimize the RMSD between two structure
fn kabsch(p: &[Vector3<f64>], q: &[Vector3<f64>]) -> Matrix3<f64> {
    let mut covariance = Matrix3::zeros();
    for (p_vec, q_vec) in p.iter().zip(q.iter()) {
        covariance += q_vec * p_vec.transpose();
    }
    let svd = covariance.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let d = if (u * v_t).determinant() < 0.0 {
        let mut correction = Matrix3::identity();
        let smallest_sigma_index = svd
            .singular_values
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap_or(2);
        correction[(smallest_sigma_index, smallest_sigma_index)] = -1.0;
        correction
    } else {
        Matrix3::identity()
    };
    v_t.transpose() * d * u.transpose()
}

pub fn parse_chain_group(chain_group: &str) -> Vec<String> {
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

/// Try to get the atom coordinates of selected chains from a structure.
pub fn try_collect_atom_positions_ref(
    pdb: &PDB,
    chain_group: &str,
) -> Result<Vec<Vector3<f64>>, String> {
    let chain_ids = parse_chain_group(chain_group);

    let mut positions: Vec<Vector3<f64>> = Vec::new();
    let mut missing_chains: Vec<String> = Vec::new();

    for chain_id in chain_ids {
        let Some(chain) = pdb
            .chains()
            .find(|chain| chain.id() == chain_id.as_str())
        else {
            missing_chains.push(chain_id);
            continue;
        };

        for residue in chain.residues() {
            let mut atoms: Vec<_> = residue.atoms().collect();
            atoms.sort_by(|a, b| a.name().cmp(b.name()));
            for atom in atoms {
                let (x, y, z) = atom.pos();
                positions.push(Vector3::new(x, y, z));
            }
        }
    }

    if !missing_chains.is_empty() || positions.is_empty() {
        if missing_chains.is_empty() {
            return Err(format!("No atoms found for chain group: {chain_group}"));
        } else {
            return Err(format!(
                "Chain(s) {:?} from chain group: {chain_group} not found in one PDB",
                missing_chains
            ));
        }
    }

    Ok(positions)
}

/// Get the atom coordinates of a selected chains from a structure
pub fn collect_atom_positions_ref(pdb: &PDB, chain_group: &str) -> Vec<Vector3<f64>> {
    match try_collect_atom_positions_ref(pdb, chain_group) {
        Ok(positions) => positions,
        Err(message) => {
            println!("{message}");
            process::exit(1);
        }
    }
}

/// Apply a transformation (rotation + translation) to a structure coordinates
pub fn apply_transform_matrix(pdb: &mut PDB, rotation: Matrix3<f64>, translation: Vector3<f64>) {
    let transformation_array = [
        [
            rotation[(0, 0)],
            rotation[(0, 1)],
            rotation[(0, 2)],
            translation[0],
        ],
        [
            rotation[(1, 0)],
            rotation[(1, 1)],
            rotation[(1, 2)],
            translation[1],
        ],
        [
            rotation[(2, 0)],
            rotation[(2, 1)],
            rotation[(2, 2)],
            translation[2],
        ],
    ];
    let transformation = TransformationMatrix::from_matrix(transformation_array);
    pdb.apply_transformation(&transformation);
}

/// Apply a transformation (rotation + translation) to a structure coordinates
pub fn apply_transform(pdb: &mut PDB, rotation: Matrix3<f64>, translation: Vector3<f64>) {
    for atom in pdb.atoms_mut() {
        let (x, y, z) = atom.pos();
        let coords = Vector3::new(x, y, z);
        let new_coords = rotation * coords + translation;
        atom.set_pos((new_coords.x, new_coords.y, new_coords.z))
            .unwrap();
    }
}

pub fn sanitize_structure(pdb: &mut PDB) {
    pdb.remove_atoms_by(|atom| atom.element() == Some(&Element::H));
    pdb.full_sort();
}

/// Store the path of each structure contained inside a directory
pub fn structure_files_from_directory(directory: &str) -> io::Result<Vec<String>> {
    let mut all_structure_names = Vec::new();
    let suffixes = [".pdb", ".cif"];

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let filename = entry.file_name();
            let filename_str = filename.to_string_lossy();
            if suffixes.iter().any(|suffix| filename_str.ends_with(suffix)) {
                all_structure_names.push(filename_str.into_owned());
            }
        }
    }

    all_structure_names.sort_by(|a, b| {
        let a_num = a
            .split('_')
            .nth(1)
            .unwrap_or("0")
            .parse::<i32>()
            .unwrap_or(0);
        let b_num = b
            .split('_')
            .nth(1)
            .unwrap_or("0")
            .parse::<i32>()
            .unwrap_or(0);
        a_num.cmp(&b_num)
    });

    Ok(all_structure_names)
}

/// Align all structures to a reference on a reference chain
pub fn all_alignment(
    filenames: &[String],
    reference_structure: &PDB,
    reference_chain: &str,
    source_path: &str,
    destination_path: &str,
    transformation_method: &str,
) {
    println!("Computing structural alignment on single thread....");
    let start = Instant::now();
    let style = default_progress_style();

    filenames
        .iter()
        .progress_with_style(style)
        .for_each(|pdb_to_align| {
            let input_name = format!("{source_path}/{pdb_to_align}");
            let (mut pdb2, _errors) = pdbtbx::open(&input_name).expect("Failed to open second PDB");
            sanitize_structure(&mut pdb2);
            align_structures_ref(
                reference_structure,
                &mut pdb2,
                reference_chain,
                transformation_method,
            );
            let output_name = format!("{destination_path}/{pdb_to_align}");
            save(&pdb2, output_name, pdbtbx::StrictnessLevel::Loose)
                .expect("Failed to save aligned PDB");
        });

    let duration = start.elapsed();
    println!("Took {:?}\n", duration);
}

/// Use multithreading to align all structures to a reference on a reference chain
pub fn parallel_all_alignment(
    filenames: &[String],
    reference_structure: &PDB,
    reference_chain: &str,
    source_path: &str,
    destination_path: &str,
    transformation_method: &str,
) {
    println!("Computing structural alignment in parallel....");
    let start = Instant::now();
    let style = default_progress_style();

    filenames
        .par_iter()
        .progress_with_style(style)
        .for_each(|pdb_to_align| {
            let input_name = format!("{source_path}/{pdb_to_align}");
            let (mut pdb2, _errors) = pdbtbx::open(&input_name).expect("Failed to open second PDB");
            sanitize_structure(&mut pdb2);
            align_structures_ref(
                reference_structure,
                &mut pdb2,
                reference_chain,
                transformation_method,
            );
            let output_name = format!("{destination_path}/{pdb_to_align}");
            save(&pdb2, output_name, pdbtbx::StrictnessLevel::Loose)
                .expect("Failed to save aligned PDB");
        });

    let duration = start.elapsed();
    println!("Took {:?}\n", duration);
}
