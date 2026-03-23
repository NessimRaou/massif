use std::time::Instant;
use indicatif::{ParallelProgressIterator};
use rayon::prelude::*;
use crate::progress::default_progress_style;
use dockq_rs::{DockQInterfaceContactsResult, DockQContacts, DockQConfig, DockQPartners, extract_contacts_from_files, all_interfaces_contacts_from_files};

fn get_contacts(pdb_file: &str, receptor: &str, ligand: &str) -> DockQContacts {
    let (_pdb, _errors) = pdbtbx::open(pdb_file).expect("Failed to open PDB");
    let contacts = extract_contacts_from_files(
      pdb_file,
      pdb_file,
      &DockQPartners {
        receptor: receptor.into(),
        ligand: ligand.into(),
      },
      &DockQConfig::default(),
    );
    contacts.expect(&format!("Error while computing {pdb_file} contacts."))
}

/// Get all atom-atom contacts from a set of structures
pub fn all_contacts(
  pdb_file_names: &[String],
  input_dir: &str
) -> Vec<Vec<DockQInterfaceContactsResult>> {
    println!("Computing contacts in the structures...");
    let start = Instant::now();
    let style = default_progress_style();

    let duration = start.elapsed();
    println!("Took {:?}\n", duration);

    let contacts: Vec<Vec<DockQInterfaceContactsResult>> = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|pdb| {
          all_interfaces_contacts_from_files(
            &format!("{input_dir}/{pdb}"),
            &format!("{input_dir}/{pdb}"),
            &DockQConfig::default()).expect("Issue computing contacts.")
          }
        ).collect();
    contacts
}

/// Compute the number of clashes from structure-extracted contacts
pub fn count_clashes(all_contacts: Vec<DockQContacts>) -> Vec<f64> {
  let pdb_clashes: Vec<f64> = all_contacts
    .par_iter()
    .map(|pdb_contacts| {pdb_contacts.clashes.len() as f64} )
    .collect();

  pdb_clashes
}

/// Evaluate, as per CAPRI criteria, the threshold of clashes to exclude structures (mean + 2σ).
pub fn clashes_threshold(pdb_clashes: &[f64]) -> f64 {
    let sum: f64 = pdb_clashes.iter().sum();
    let mean = sum / pdb_clashes.len() as f64;
    let variance: f64 = pdb_clashes.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (pdb_clashes.len() as f64 - 1.0);
    let std_dev = variance.sqrt();
    mean + 2.0 * std_dev
}
