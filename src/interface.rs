use std::collections::HashMap;
use std::time::Instant;

use indicatif::{ParallelProgressIterator};
use nalgebra::Point3;
use pdbtbx::ContainsAtomConformer;
use pdbtbx::ContainsAtomConformerResidue;
use pdbtbx::ContainsAtomConformerResidueChain;
use pdbtbx::{AtomConformerResidueChainModel};

use rayon::prelude::*;

use crate::progress::default_progress_style;

/// Process inspired from Pierce Lab
/// https://github.com/piercelab/alphafold_v2.2_customize/blob/master/get_interface_plddt.pl
/// Compute the plddt at the user-specified interface
pub fn compute_interface_plddt(
    pdb_file: &str,
    aggregate_1: &str,
    aggregate_2: &str,
    threshold: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (pdb, _errors) = pdbtbx::open(pdb_file).expect("Failed to open PDB");

    let mut aggregate_1_atoms = Vec::new();
    let mut aggregate_2_atoms = Vec::new();
    for atom in pdb.atoms_with_hierarchy() {
        let chain_id = atom.chain().id();
        if aggregate_1.contains(chain_id) {
            aggregate_1_atoms.push(atom);
        } else if aggregate_2.contains(chain_id) {
            aggregate_2_atoms.push(atom);
        }
    }

    let get_residue_key = |atom: &AtomConformerResidueChainModel| -> Option<String> {
        if atom.atom().name() == "H" {
            return None;
        }
        let chain_id = atom.chain().id();
        let residue = atom.residue();
        let serial = residue.serial_number();
        let res_num = match residue.insertion_code() {
            Some(code) => format!("{}{}", serial, code),
            None => serial.to_string(),
        };
        let res_name = residue
            .name()
            .expect("Missing residue name")
            .trim()
            .to_string();
        Some(format!("{res_num}\t{chain_id}\t{res_name}"))
    };

    let mut aggregate_1_residues = HashMap::new();
    let mut aggregate_2_residues = HashMap::new();

    for atom2 in &aggregate_2_atoms {
        let key2 = match get_residue_key(atom2) {
            Some(key) => key,
            None => continue,
        };
        let pos2 = Point3::new(atom2.atom().x(), atom2.atom().y(), atom2.atom().z());
        for atom1 in &aggregate_1_atoms {
            let pos1 = Point3::new(atom1.atom().x(), atom1.atom().y(), atom1.atom().z());
            if nalgebra::distance(&pos1, &pos2) >= threshold {
                continue;
            }
            let key1 = match get_residue_key(atom1) {
                Some(key) => key,
                None => continue,
            };
            aggregate_1_residues.insert(key1, atom1.atom().b_factor());
            aggregate_2_residues.insert(key2.clone(), atom2.atom().b_factor());
        }
    }

    let total_count = aggregate_1_residues.len() + aggregate_2_residues.len();
    if total_count == 0 {
        return Err("No residues found at the interface".into());
    }
    let total_plddt: f64 = aggregate_1_residues
        .values()
        .chain(aggregate_2_residues.values())
        .sum();

    Ok(total_plddt / total_count as f64)
}

/// Compute the iplddt for a set a structures with the same chain ids
pub fn all_iplddt(
    input_dir: &str,
    pdb_file_names: &[String],
    aggregate_1: &str,
    aggregate_2: &str,
    threshold: f64,
) -> Vec<f64> {
    println!("Computing i-plddt on {} structures", pdb_file_names.len());
    let start = Instant::now();
    let style = default_progress_style();

    let all_iplddt: Vec<f64> = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|filename| {
            let pdb_file = format!("{input_dir}/{filename}");
            compute_interface_plddt(&pdb_file, aggregate_1, aggregate_2, threshold).unwrap_or(-1.0)
        })
        .collect();

    let duration = start.elapsed();
    println!("Took {:?}\n", duration);
    all_iplddt
}
