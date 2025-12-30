use std::time::Instant;

use csv::Writer;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use nalgebra::Vector3;
use pdbtbx::ContainsAtomConformer;
use pdbtbx::PDB;
use rayon::prelude::*;

use crate::progress::default_progress_style;

/// Compute TM score between two structure
fn tm_score(pdb1: &PDB, pdb2: &PDB) -> f64 {
    let l = pdb1.residue_count() as f64;
    let d0 = 1.24 * f64::cbrt(l - 15.0) - 1.8;
    let pdb1_ca_coord = get_alpha_carbon_coords(pdb1);
    let pdb2_ca_coord = get_alpha_carbon_coords(pdb2);
    let tm_score_sum: f64 = pdb1_ca_coord
        .iter()
        .zip(pdb2_ca_coord.iter())
        .map(|(coord1, coord2)| {
            let diff = coord1 - coord2;
            let squared_distance = diff.dot(&diff);
            1.0 / (1.0 + squared_distance / d0.powi(2))
        })
        .sum();
    tm_score_sum / l
}

/// Compute RMSD between two structure
fn rmsd(pdb1: &PDB, pdb2: &PDB) -> f64 {
    let pdb1_coord = get_atom_coordinates(pdb1);
    let pdb2_coord = get_atom_coordinates(pdb2);
    let rmsd_sum: f64 = pdb1_coord
        .par_iter()
        .zip(pdb2_coord.par_iter())
        .map(|(coord1, coord2)| {
            let diff = coord1 - coord2;
            diff.dot(&diff)
        })
        .sum();
    (rmsd_sum / pdb1_coord.len() as f64).sqrt()
}

/// Compute the distance between two structure with different methods (RMSD, TMscore...)
fn compute_distance(pdb1: &PDB, pdb2: &PDB, mode: &str) -> f64 {
    match mode {
        "rmsd-cur" => rmsd(pdb1, pdb2),
        "tm-score" => tm_score(pdb1, pdb2),
        _ => {
            println!("No '{mode}' available");
            0.0
        }
    }
}

/// Get all atom's coordinates from a given structure
fn get_atom_coordinates(pdb: &PDB) -> Vec<Vector3<f64>> {
    pdb.atoms()
        .map(|atom| {
            let pos = atom.pos();
            Vector3::new(pos.0, pos.1, pos.2)
        })
        .collect()
}

/// Get the coordinates of all alpha carbons
fn get_alpha_carbon_coords(pdb: &PDB) -> Vec<Vector3<f64>> {
    let all_alpha_carbon = pdb.find(pdbtbx::Search::Single(pdbtbx::Term::AtomName(
        "CA".to_owned(),
    )));
    all_alpha_carbon
        .map(|ca| {
            let positions = ca.atom().pos();
            Vector3::new(positions.0, positions.1, positions.2)
        })
        .collect()
}

/// Record the results in a CSV file
fn save_csv(pdbs: &[String], pdb_ref: &str, dists: &[f64], csv_name: &str) {
    let mut csv_write = Writer::from_path(csv_name).expect("Unable to create csv file");
    csv_write
        .write_record(["reference", "target", "distance"])
        .expect("Unable to write csv header");
    for (target, dist) in pdbs.iter().zip(dists.iter()) {
        csv_write
            .write_record([pdb_ref, target, &dist.to_string()])
            .expect("Unable to write csv row");
    }
}

/// Compute the distance (RMSD, TMscore...) between a reference structure
/// and each structure of a set
pub fn all_distances(
    pdb_ref_file: &str,
    pdb_file_names: &[String],
    source_path: &str,
    distance_mode: &str,
) -> Vec<f64> {
    println!(
        "Computing {distance_mode} between ref & {} structures\nReference: {}",
        pdb_file_names.len(),
        pdb_ref_file
    );
    let start = Instant::now();
    let style = default_progress_style();
    let (pdb1, _errors) = pdbtbx::open(pdb_ref_file).expect("Failed to open reference PDB");

    let all_distances: Vec<f64> = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|pdb_to_compare| {
            let pdb_file = format!("{source_path}/{pdb_to_compare}");
            let (pdb2, _errors) = pdbtbx::open(&pdb_file).expect("Failed to open second PDB");
            compute_distance(&pdb1, &pdb2, distance_mode)
        })
        .collect();

    let csv_path = format!("./{}_{}.csv", distance_mode, pdb_file_names.len());
    save_csv(pdb_file_names, pdb_ref_file, &all_distances, &csv_path);

    let duration = start.elapsed();
    println!("Took {:?}\n", duration);
    all_distances
}
