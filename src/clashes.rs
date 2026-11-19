use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use indicatif::{ParallelProgressIterator, ProgressIterator};
use pdbtbx::{Element, PDB};
use rayon::prelude::*;

use crate::alignment::structure_file_path;
use crate::contacts::CLASH_CUTOFF;
use crate::progress::default_progress_style;
use dockq_rs::{all_interfaces_contacts, DockQConfig};

/// Atoms whose pairs closer than `CLASH_CUTOFF` define the backbone overlaps
const BACKBONE: [&str; 4] = ["N", "CA", "C", "O"];
/// Minimal residue-number separation of an intra-chain clash
const MIN_SEPARATION: isize = 3;

/// Clash measures of one chain of a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainClashes {
    pub chain: String,
    /// Residues of the chain with at least one heavy atom.
    pub residues: usize,
    /// Inter-chain clashing residue pairs involving the chain (the `nclash` pairs it takes part in).
    pub nclash: usize,
    /// Overlap ratio of the chain with all the other chains: its residues with a backbone atom closer than `CLASH_CUTOFF` to the backbone of any
    /// other chain (each counted once, whatever the number of chains it overlaps), divided by the chain's residues.
    pub overlap: f64,
    /// Residue pairs of the chain, at least `MIN_SEPARATION` apart, with two heavy atoms closer than `CLASH_CUTOFF`.
    pub intra: usize,
    /// Share of the chain's residues in an intra-chain clash between backbone atoms (a collapsed chain).
    pub collapse: f64,
}

/// Clash measures of one model: the inter-chain clashing residue pairs  and the chains' measures, in the order of the model's chains.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelClashes {
    pub nclash: usize,
    pub chains: Vec<ChainClashes>,
}

struct GridAtom {
    chain_index: usize,
    residue_index: usize,
    serial_number: isize,
    backbone: bool,
    x: f64,
    y: f64,
    z: f64,
}

fn is_eligible_atom(atom: &pdbtbx::Atom) -> bool {
    !atom.hetero() && atom.element() != Some(&Element::H)
}

fn grid_cell(x: f64, y: f64, z: f64) -> (i64, i64, i64) {
    let inverse_cell_size = 1.0 / CLASH_CUTOFF;
    (
        (x * inverse_cell_size).floor() as i64,
        (y * inverse_cell_size).floor() as i64,
        (z * inverse_cell_size).floor() as i64,
    )
}

/// The clash measures of a structure's first model. Heavy atoms of ATOM records only. A residue is one pdbtbx residue.
pub fn structure_clashes(pdb: &PDB) -> Result<ModelClashes, String> {
    let Some(model) = pdb.model(0) else {
        return Err(String::from("the structure has no model"));
    };
    let chain_ids: Vec<String> = model.chains().map(|chain| chain.id().to_string()).collect();

    let mut nclash_per_chain: HashMap<String, usize> = HashMap::new();
    let mut nclash = 0;
    for interface in all_interfaces_contacts(pdb, &DockQConfig::default()) {
        let contacts = interface.result.map_err(|err| {
            format!("contacts of {}:{} failed: {}", interface.partners.receptor, interface.partners.ligand, err)
        })?;
        let count = contacts.iter().filter(|contact| contact.distance < CLASH_CUTOFF).count();
        nclash += count;
        *nclash_per_chain.entry(interface.partners.receptor).or_default() += count;
        *nclash_per_chain.entry(interface.partners.ligand).or_default() += count;
    }

    let mut residues = vec![0_usize; chain_ids.len()];
    let mut grid: HashMap<(i64, i64, i64), Vec<GridAtom>> = HashMap::new();
    let mut intra_pairs: Vec<HashSet<(usize, usize)>> = vec![HashSet::new(); chain_ids.len()];
    let mut collapsed: Vec<HashSet<usize>> = vec![HashSet::new(); chain_ids.len()];
    let mut overlapping: Vec<HashSet<usize>> = vec![HashSet::new(); chain_ids.len()];
    let squared_cutoff = CLASH_CUTOFF * CLASH_CUTOFF;
    let mut residue_index = 0;

    for (chain_index, chain) in model.chains().enumerate() {
        for residue in chain.residues() {
            let serial_number = residue.serial_number();
            let mut has_atoms = false;
            for atom in residue.atoms() {
                if !is_eligible_atom(atom) {
                    continue;
                }
                has_atoms = true;
                let (x, y, z) = atom.pos();
                let backbone = BACKBONE.contains(&atom.name());
                let (cx, cy, cz) = grid_cell(x, y, z);
                for dx in -1_i64..=1 {
                    for dy in -1_i64..=1 {
                        for dz in -1_i64..=1 {
                            let Some(neighbours) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                                continue;
                            };
                            for other in neighbours {
                                let (ox, oy, oz) = (other.x - x, other.y - y, other.z - z);
                                if ox * ox + oy * oy + oz * oz >= squared_cutoff {
                                    continue;
                                }
                                if other.chain_index == chain_index {
                                    if (other.serial_number - serial_number).abs() < MIN_SEPARATION {
                                        continue;
                                    }
                                    let pair = (other.residue_index.min(residue_index), other.residue_index.max(residue_index));
                                    intra_pairs[chain_index].insert(pair);
                                    if backbone && other.backbone {
                                        collapsed[chain_index].insert(other.residue_index);
                                        collapsed[chain_index].insert(residue_index);
                                    }
                                } else if backbone && other.backbone {
                                    overlapping[chain_index].insert(residue_index);
                                    overlapping[other.chain_index].insert(other.residue_index);
                                }
                            }
                        }
                    }
                }
                grid.entry((cx, cy, cz)).or_default().push(GridAtom {
                    chain_index,
                    residue_index,
                    serial_number,
                    backbone,
                    x,
                    y,
                    z,
                });
            }
            if has_atoms {
                residues[chain_index] += 1;
            }
            residue_index += 1;
        }
    }

    let share = |count: usize, chain_index: usize| {
        if residues[chain_index] == 0 {
            0.0
        } else {
            count as f64 / residues[chain_index] as f64
        }
    };
    let chains = chain_ids
        .iter()
        .enumerate()
        .map(|(chain_index, chain)| ChainClashes {
            chain: chain.clone(),
            residues: residues[chain_index],
            nclash: nclash_per_chain.get(chain).copied().unwrap_or(0),
            overlap: share(overlapping[chain_index].len(), chain_index),
            intra: intra_pairs[chain_index].len(),
            collapse: share(collapsed[chain_index].len(), chain_index),
        })
        .collect();

    Ok(ModelClashes { nclash, chains })
}

fn structure_clashes_from_path(pdb_path: &str) -> Result<ModelClashes, String> {
    // pdbtbx panics on some malformed records instead of returning an error.
    let opened = std::panic::catch_unwind(|| pdbtbx::open(pdb_path)).map_err(|panic| {
        let reason = panic
            .downcast_ref::<&str>()
            .map(|reason| reason.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        format!("failed to read {pdb_path}: {reason}")
    })?;
    let (structure, _errors) = opened.map_err(|errors| format!("failed to open {pdb_path}: {errors:?}"))?;
    structure_clashes(&structure).map_err(|err| format!("{pdb_path}: {err}"))
}

/// The clash measures of a set of structures, in the order of `pdb_file_names`; the first structure that fails gives the error.
pub fn all_structure_clashes(pdb_file_names: &[String], input_dir: &str, parallel: bool) -> Result<Vec<ModelClashes>, String> {
    println!("Computing clashes in the structures...");
    let start = Instant::now();
    let style = default_progress_style();

    let clashes: Result<Vec<ModelClashes>, String> = if parallel {
        pdb_file_names
            .par_iter()
            .progress_with_style(style)
            .map(|pdb| structure_clashes_from_path(&structure_file_path(input_dir, pdb)))
            .collect()
    } else {
        pdb_file_names
            .iter()
            .progress_with_style(style)
            .map(|pdb| structure_clashes_from_path(&structure_file_path(input_dir, pdb)))
            .collect()
    };

    println!("Took {:?}\n", start.elapsed());
    clashes
}
