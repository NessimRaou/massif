use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressIterator};
use pdbtbx::ContainsAtomConformer;
use pdbtbx::ContainsAtomConformerResidue;
use pdbtbx::ContainsAtomConformerResidueChain;
use pdbtbx::{Element, PDB};
use rayon::prelude::*;

use crate::progress::default_progress_style;

#[derive(Clone, Debug)]
struct ContactAtom {
    atom: pdbtbx::Atom,
    residue: isize,
    chain_id: String,
}

#[derive(Clone, Debug)]
pub struct Contact {
    pub(crate) distance: f64,
    atom1: ContactAtom,
    atom2: ContactAtom,
    pub contact_type: String,
}

/// Get all atom-atom contacts from a structure
fn get_contacts(pdb_file: &str) -> Vec<Contact> {
    let (pdb, _errors) = pdbtbx::open(pdb_file).expect("Failed to open PDB");
    let rtree = pdb.create_hierarchy_rtree();
    let atoms = pdb.atoms_with_hierarchy();

    let atoms_info: Vec<ContactAtom> = atoms
        .map(|atom| ContactAtom {
            atom: atom.atom().to_owned(),
            residue: atom.residue().serial_number(),
            chain_id: atom.chain().id().to_string(),
        })
        .collect();

    let mut all_contact: Vec<Contact> = Vec::new();
    for atom in atoms_info {
        let contacts = rtree
            .locate_within_distance(atom.atom.pos(), 10.0)
            .map(|contact| ContactAtom {
                atom: contact.atom().to_owned(),
                residue: contact.residue().serial_number(),
                chain_id: contact.chain().id().to_string(),
            })
            .collect::<Vec<ContactAtom>>();

        if contacts.is_empty() {
            continue;
        }

        for atom_contact in contacts {
            let both_hydrogens = (atom.atom.element() == Some(&Element::H))
                & (atom_contact.atom.element() == Some(&Element::H));
            let is_in_same_chain = atom.chain_id == atom_contact.chain_id;
            if both_hydrogens | is_in_same_chain {
                continue;
            }

            let distance = atom.atom.distance(&atom_contact.atom);
            let contact_type = match distance {
                dist if dist <= 3.0 => "clash",
                dist if dist <= 5.0 => "contact",
                _ => "interface",
            }
            .to_string();

            all_contact.push(Contact {
                distance,
                atom1: atom.clone(),
                atom2: atom_contact,
                contact_type,
            });
        }
    }

    all_contact
}

/// Get all atom-atom contacts from a set of structures
pub fn all_contacts(pdb_file_names: &[String], input_dir: &str) -> Vec<Vec<Contact>> {
    println!("Computing contacts in the structures...");
    let start = Instant::now();
    let style = default_progress_style();

    let contacts: Vec<Vec<Contact>> = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|pdb| get_contacts(&format!("{input_dir}/{pdb}")))
        .collect();

    let duration = start.elapsed();
    println!("Took {:?}\n", duration);
    contacts
}

/// Compute the number of clashes from structure-extracted contacts
pub fn count_clashes(all_contacts: &[Vec<Contact>]) -> (f64, Vec<f64>) {
    let mut pdb_clashes: Vec<f64> = Vec::new();
    for pdb_contacts in all_contacts {
        let mut n_clashes = 0.0;
        pdb_contacts
            .iter()
            .for_each(|contact| match contact.contact_type.as_str() {
                "clash" => n_clashes += 0.5,
                _ => (),
            });
        pdb_clashes.push(n_clashes);
    }
    (clashes_threshold(&pdb_clashes), pdb_clashes)
}

/// Evaluate, as per CAPRI criteria, the thresholds of clashes to exclude structure
fn clashes_threshold(pdb_clashes: &[f64]) -> f64 {
    let sum: f64 = pdb_clashes.iter().sum();
    let mean = sum / pdb_clashes.len() as f64;
    let variance: f64 = pdb_clashes.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (pdb_clashes.len() as f64 - 1.0);
    let std_dev = variance.sqrt();
    mean + 2.0 * std_dev
}
