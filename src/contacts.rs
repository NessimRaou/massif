use std::io::Write;
use std::path::Path;
use std::time::Instant;

use csv::Writer;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use serde::Serialize;

use crate::progress::default_progress_style;
use dockq_rs::{
    all_interfaces_contacts_from_files,
    extract_contacts_from_files,
    DockQAlignedContact,
    DockQConfig,
    DockQContacts,
    DockQInterfaceContactsResult,
    DockQPartners,
    ResidueKey,
};

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
    input_dir: &str,
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
                &DockQConfig::default(),
            )
            .expect("Issue computing contacts.")
        })
        .collect();
    contacts
}

/// Compute the number of clashes from structure-extracted contacts
pub fn count_clashes(all_contacts: Vec<DockQContacts>) -> Vec<f64> {
    let pdb_clashes: Vec<f64> = all_contacts
        .par_iter()
        .map(|pdb_contacts| pdb_contacts.clashes.len() as f64)
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

// --- Contact detail CSV (tidy export) ------------------------------------

/// Format a [`ResidueKey`] the same way as DockQ residue labels in CSV exports.
fn residue_key_format(key: &ResidueKey) -> String {
    let ins = key.insertion_code.as_deref().unwrap_or("");
    format!(
        "{}:{}:{}{}",
        key.chain_id, key.residue_name, key.residue_number, ins
    )
}

#[derive(Serialize)]
pub struct FlatContactRecord<'a> {
    pub receptor: &'a str,
    pub ligand: &'a str,
    pub contact_category: &'a str,
    pub rec_native: String,
    pub lig_native: String,
    pub rec_model: String,
    pub lig_model: String,
}

fn write_aligned<W: Write>(
    wtr: &mut Writer<W>,
    rec_chain: &str,
    lig_chain: &str,
    category: &str,
    list: &[DockQAlignedContact],
) -> Result<(), csv::Error> {
    for contact in list {
        wtr.serialize(FlatContactRecord {
            receptor: rec_chain,
            ligand: lig_chain,
            contact_category: category,
            rec_native: residue_key_format(&contact.receptor_native),
            lig_native: residue_key_format(&contact.ligand_native),
            rec_model: residue_key_format(&contact.receptor_model),
            lig_model: residue_key_format(&contact.ligand_model),
        })?;
    }
    Ok(())
}

fn write_all_rows<W: Write>(
    wtr: &mut Writer<W>,
    results: &[DockQInterfaceContactsResult],
) -> Result<(), csv::Error> {
    for res in results {
        let rec_chain = res.partners.receptor.as_str();
        let lig_chain = res.partners.ligand.as_str();

        if let Ok(contacts) = &res.result {
            write_aligned(wtr, rec_chain, lig_chain, "shared", &contacts.shared_contacts)?;
            write_aligned(wtr, rec_chain, lig_chain, "nonnative", &contacts.nonnative_contacts)?;
            write_aligned(wtr, rec_chain, lig_chain, "clash", &contacts.clashes)?;
            write_aligned(wtr, rec_chain, lig_chain, "model_only", &contacts.model_contacts)?;

            for contact in &contacts.native_contacts {
                wtr.serialize(FlatContactRecord {
                    receptor: rec_chain,
                    ligand: lig_chain,
                    contact_category: "native_only",
                    rec_native: residue_key_format(&contact.receptor),
                    lig_native: residue_key_format(&contact.ligand),
                    rec_model: "-".to_string(),
                    lig_model: "-".to_string(),
                })?;
            }
        }
    }
    Ok(())
}

/// Serializes interface contact breakdown to a CSV string (successful interfaces only).
pub fn results_to_csv_string(
    results: &[DockQInterfaceContactsResult],
) -> Result<String, csv::Error> {
    let mut wtr = Writer::from_writer(vec![]);
    write_all_rows(&mut wtr, results)?;
    let bytes = wtr.into_inner().map_err(|e| e.into_error())?;
    String::from_utf8(bytes).map_err(|e| {
        csv::Error::from(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

/// Writes the same content as [`results_to_csv_string`] to `path`.
pub fn write_interface_contacts_csv(
    path: &Path,
    results: &[DockQInterfaceContactsResult],
) -> Result<(), csv::Error> {
    let mut wtr = Writer::from_path(path)?;
    write_all_rows(&mut wtr, results)?;
    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod contact_csv_tests {
    use super::*;

    #[test]
    fn empty_results_to_csv_string() {
        let s = results_to_csv_string(&[]).unwrap();
        assert!(s.is_empty() || s.lines().count() <= 1);
    }
}
