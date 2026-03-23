use std::io::Write;
use std::path::Path;
use std::time::Instant;

use csv::Writer;
use indicatif::ParallelProgressIterator;
use pdbtbx::PDB;
use rayon::prelude::*;
use serde::Serialize;

use crate::progress::default_progress_style;
use dockq_rs::{
    all_interfaces_contacts, compare_contacts, DockQConfig, DockQContact, DockQContacts,
    DockQError, DockQInterfaceContactsResult, DockQPartners, ResidueKey,
};

#[derive(Clone, Debug)]
pub struct ModelContactsSummary {
    pub interfaces: Vec<DockQInterfaceContactsResult>,
    pub clashes: f64,
}

fn read_structure(pdb_path: &str) -> PDB {
    let (structure, _errors) = pdbtbx::open(pdb_path)
        .unwrap_or_else(|_| panic!("Failed to open structure file {pdb_path}"));
    structure
}

fn clash_count_for_interface(
    structure: &PDB,
    partners: &DockQPartners,
    config: &DockQConfig,
    pdb_path: &str,
) -> f64 {
    match compare_contacts(structure, structure, partners, config) {
        Ok(contacts) => contacts.clashes.len() as f64,
        Err(DockQError::NoNativeContacts) => 0.0,
        Err(err) => panic!(
            "Issue computing clashes for interface {}:{} in {}: {}",
            partners.receptor, partners.ligand, pdb_path, err
        ),
    }
}

fn structure_interfaces_from_path(
    pdb_path: &str,
    config: &DockQConfig,
) -> Vec<DockQInterfaceContactsResult> {
    let structure = read_structure(pdb_path);
    all_interfaces_contacts(&structure, config)
}

fn structure_contacts_summary(pdb_path: &str, config: &DockQConfig) -> ModelContactsSummary {
    let structure = read_structure(pdb_path);
    let interfaces = all_interfaces_contacts(&structure, config);
    let clashes = interfaces
        .iter()
        .map(|interface| {
            clash_count_for_interface(&structure, &interface.partners, config, pdb_path)
        })
        .sum();

    ModelContactsSummary {
        interfaces,
        clashes,
    }
}

/// Get all residue-residue contacts and clash counts from a set of structures.
pub fn all_contacts_with_clashes(
    pdb_file_names: &[String],
    input_dir: &str,
) -> Vec<ModelContactsSummary> {
    println!("Computing contacts in the structures...");
    let start = Instant::now();
    let style = default_progress_style();

    let summaries = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|pdb| {
            structure_contacts_summary(&format!("{input_dir}/{pdb}"), &DockQConfig::default())
        })
        .collect();

    println!("Took {:?}\n", start.elapsed());
    summaries
}

/// Get all residue-residue contacts from a set of structures.
pub fn all_contacts(
    pdb_file_names: &[String],
    input_dir: &str,
) -> Vec<Vec<DockQInterfaceContactsResult>> {
    println!("Computing contacts in the structures...");
    let start = Instant::now();
    let style = default_progress_style();

    let contacts = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|pdb| {
            structure_interfaces_from_path(&format!("{input_dir}/{pdb}"), &DockQConfig::default())
        })
        .collect();

    println!("Took {:?}\n", start.elapsed());
    contacts
}

/// Compute the total number of clashes per structure.
pub fn all_clashes(pdb_file_names: &[String], input_dir: &str) -> Vec<f64> {
    all_contacts_with_clashes(pdb_file_names, input_dir)
        .into_iter()
        .map(|summary| summary.clashes)
        .collect()
}

/// Compute the number of clashes from native/model DockQ contact comparisons.
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
    pub rec_contact: String,
    pub lig_contact: String,
}

fn write_contacts<W: Write>(
    wtr: &mut Writer<W>,
    rec_chain: &str,
    lig_chain: &str,
    list: &[DockQContact],
) -> Result<(), csv::Error> {
    for contact in list {
        wtr.serialize(FlatContactRecord {
            receptor: rec_chain,
            ligand: lig_chain,
            rec_contact: residue_key_format(&contact.receptor),
            lig_contact: residue_key_format(&contact.ligand),
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
            write_contacts(wtr, rec_chain, lig_chain, contacts)?;
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
