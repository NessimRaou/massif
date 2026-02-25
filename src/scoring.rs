use std::time::Instant;

use indicatif::{ParallelProgressIterator};
use rayon::prelude::*;

use crate::progress::default_progress_style;

/// Compute a score for a structure interface likelihood
/// /!\ placeholder code, not implemented yet
pub fn score_interface(pdb_file_names: &[String], _input_dir: &str, ptm_type: &str) -> Vec<f64> {
    println!(
        "Scoring interfaces ({ptm_type}) for {} structures",
        pdb_file_names.len()
    );
    let start = Instant::now();
    let style = default_progress_style();

    let predicted_tms: Vec<f64> = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|_pdb| match ptm_type {
            "pTM" | "ipTM" | "pITM" => 0.0,
            _ => {
                println!("No transformation method called \"{ptm_type}\"");
                0.0
            }
        })
        .collect();

    let duration = start.elapsed();
    println!("Took {:?}\n", duration);
    predicted_tms
}

/// Compute a 'score' for a set of given structure
pub fn all_scores_computation(input_dir: &str, pdb_file_names: &[String]) -> Vec<f64> {
    println!("Computing score on {} structures", pdb_file_names.len());
    let start = Instant::now();
    let style = default_progress_style();

    let all_scores: Vec<f64> = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|filename| {
            let pdb_file = format!("{input_dir}/{filename}");
            score_structure(&pdb_file)
        })
        .collect();

    let duration = start.elapsed();
    println!("Took {:?}\n", duration);
    all_scores
}

/// Compute a 'score' for a set of given structure
fn score_structure(_structure_path: &str) -> f64 {
    1.0
}
