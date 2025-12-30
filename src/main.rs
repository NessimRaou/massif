use clap::{Parser, Subcommand};
use csv::{Reader, Writer};
use massif::*;
use indexmap::{IndexMap, IndexSet};
use pdbtbx::open;
use std::{
    error::Error,
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
};

type StructuredRows = IndexMap<String, IndexMap<String, String>>;

fn structured_csv_path(csv_filename: &str) -> String {
    let path = Path::new(csv_filename);
    let mut new_name = path
        .file_stem()
        .map(|stem| format!("{}_alternative", stem.to_string_lossy()))
        .unwrap_or_else(|| String::from("alternative_output"));
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        new_name.push('.');
        new_name.push_str(ext);
    }
    let mut new_path = PathBuf::from(path);
    new_path.set_file_name(new_name);
    new_path.to_string_lossy().into_owned()
}

fn load_structured_rows(path: &str) -> Result<(StructuredRows, IndexSet<String>), Box<dyn Error>> {
    let mut rows = StructuredRows::new();
    let mut header_order = IndexSet::new();
    if !Path::new(path).exists() {
        return Ok((rows, header_order));
    }
    let mut rdr = Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    for header in headers.iter() {
        header_order.insert(header.to_string());
    }
    let models_idx = headers.iter().position(|h| h == "Models").unwrap_or(0);
    for result in rdr.records() {
        let record = result?;
        let model_value = record.get(models_idx).unwrap_or("").to_string();
        if model_value.is_empty() {
            continue;
        }
        let entry = rows
            .entry(model_value.clone())
            .or_insert_with(IndexMap::new);
        for (idx, header) in headers.iter().enumerate() {
            let value = record.get(idx).unwrap_or("").to_string();
            entry.insert(header.to_string(), value);
        }
    }
    Ok((rows, header_order))
}

fn columns_to_structured_rows(
    headers: &[String],
    columns: &[Vec<String>],
) -> Result<(StructuredRows, IndexSet<String>), Box<dyn Error>> {
    if headers.is_empty() {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidData,
            "No headers provided",
        )));
    }
    if headers.len() != columns.len() {
        eprintln!(
      "Structured CSV: header/data count mismatch (headers: {}, columns: {}); extra columns will be ignored",
      headers.len(),
      columns.len()
    );
    }
    let mut header_order = IndexSet::new();
    for header in headers {
        header_order.insert(header.clone());
    }
    let models_idx = headers.iter().position(|h| h == "Models").ok_or_else(|| {
        Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Missing 'Models' column",
        )) as Box<dyn Error>
    })?;
    if columns.len() <= models_idx {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Missing 'Models' column data",
        )));
    }
    let row_count = columns[models_idx].len();
    let mut rows = StructuredRows::new();
    for row_idx in 0..row_count {
        let model_value = columns[models_idx]
            .get(row_idx)
            .cloned()
            .unwrap_or_default();
        if model_value.is_empty() {
            continue;
        }
        let entry = rows
            .entry(model_value.clone())
            .or_insert_with(IndexMap::new);
        entry.insert(String::from("Models"), model_value.clone());
        for (col_idx, header) in headers.iter().enumerate() {
            if let Some(column) = columns.get(col_idx) {
                let value = column.get(row_idx).cloned().unwrap_or_default();
                entry.insert(header.clone(), value);
            }
        }
    }
    Ok((rows, header_order))
}

fn merge_structured_rows(existing: &mut StructuredRows, incoming: StructuredRows) {
    for (model, metrics) in incoming {
        let entry = existing.entry(model).or_insert_with(IndexMap::new);
        for (key, value) in metrics {
            entry.insert(key, value);
        }
    }
}

fn write_structured_csv(
    path: &str,
    rows: &StructuredRows,
    header_order: &IndexSet<String>,
) -> Result<(), Box<dyn Error>> {
    let mut headers: Vec<String> = Vec::new();
    if header_order.contains("Models") {
        headers.push(String::from("Models"));
    }
    for header in header_order.iter() {
        if header == "Models" {
            continue;
        }
        headers.push(header.clone());
    }
    if !headers.iter().any(|h| h == "Models") {
        headers.insert(0, String::from("Models"));
    }
    let mut wtr = Writer::from_path(path)?;
    wtr.write_record(&headers)?;
    for (model, metrics) in rows.iter() {
        let mut row: Vec<String> = Vec::with_capacity(headers.len());
        for header in headers.iter() {
            if header == "Models" {
                row.push(model.clone());
            } else if let Some(value) = metrics.get(header) {
                row.push(value.clone());
            } else {
                row.push(String::new());
            }
        }
        wtr.write_record(row)?;
    }
    wtr.flush()?;
    println!("Structured CSV file {} written successfully", path);
    Ok(())
}

fn report_to_csv_structured(
    csv_filename: &str,
    table: Vec<Vec<String>>,
    headers: Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let structured_path = structured_csv_path(csv_filename);
    let (mut existing_rows, mut header_order) = load_structured_rows(&structured_path)?;
    let (incoming_rows, incoming_headers) = columns_to_structured_rows(&headers, &table)?;
    for header in incoming_headers.iter() {
        header_order.insert(header.clone());
    }
    if !header_order.contains("Models") {
        header_order.insert(String::from("Models"));
    }
    merge_structured_rows(&mut existing_rows, incoming_rows);
    write_structured_csv(&structured_path, &existing_rows, &header_order)?;
    Ok(())
}

#[derive(Subcommand)]
enum Commands {
    /// Fit the structures on a given reference and computes their distance to it.
    Fit {
        /// Path to store the aligned structures
        output_dir: String,
        /// Structure to use as a reference for the alignment
        reference_structure: String,
        /// Aggregated identifiers of the chains used for the fitting (e.g. "AB" or "C")
        chain_ids: String, // Identifiers of the chains between which the distance is computed
                           // dist_to_chains: String
    },
    /// Complete analysis on contacts.
    Contacts {
        /// Path to store the aligned structures
        output_dir: String,
    },
    /// Compute the plddt at the interface between two group of chains.
    Iplddt {
        /// Aggregated identifiers of the interface's first group of chains (e.g. "AB" or "C")
        aggregate_1: String,
        /// Aggregated identifiers of the interface's second group of chains (e.g. "AB" or "C")
        aggregate_2: String,
        /// Distance threshold between two residues to count as interface
        threshold: String,
    },
    Distances {
        filename: String,
        chain_pairs: String,
    },
    Scoring {},
}

#[derive(Parser)]
#[command(version = "1.0", author = "Nessim Raouraoua")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    structure_dir: String,
    output_csv: String,
    /// Compute alignments on single thread instead of multithreaded computations
    #[arg(short, long, default_value_t = false)]
    disable_parallel: bool,
}

fn main() {
    // args parsing
    let args = Cli::parse();
    let do_in_parallel = !args.disable_parallel;
    let structure_dir = args.structure_dir;
    let file_names =
        structure_files_from_directory(&structure_dir).expect("Error reading directory {dirname}");
    let output_csv = args.output_csv;

    let mut report_colnames: Vec<String> = Vec::new();
    let mut final_report: Vec<Vec<String>> = Vec::new();
    // function calling depending on subcommand
    match args.command {
        Commands::Fit {
            output_dir,
            reference_structure,
            chain_ids,
        } => {
            //, dist_to_chains} => {
            let reference_chain = chain_ids;
            let (pdb1, _errors) = open(&reference_structure).expect("Failed to open first PDB");
            // allignment computing
            if do_in_parallel {
                parallel_all_alignment(
                    &file_names,
                    &pdb1,
                    &reference_chain,
                    &structure_dir,
                    &output_dir,
                    "per_atom",
                );
            } else {
                all_alignment(
                    &file_names,
                    &pdb1,
                    &reference_chain,
                    &structure_dir,
                    &output_dir,
                    "per_atom",
                );
            }
            // Computing of distances
            let compute_distance = true;
            if compute_distance {
                let dtype = "tm-score"; // "tm-score"|"rmsd-cur"
                let distances =
                    all_distances(&reference_structure, &file_names, &output_dir, dtype);
                let distances_string: Vec<String> =
                    distances.iter().map(|&num| num.to_string()).collect();
                report_colnames.push(format!(
                    "TM-score to {}",
                    Path::new(&reference_structure)
                        .file_stem()
                        .expect("No basename for this path")
                        .to_string_lossy()
                ));
                final_report.push(distances_string);
            }
        }
        Commands::Contacts { output_dir } => {
            // Contacts and clashes
            let contacts: Vec<Vec<Contact>> = all_contacts(&file_names, &structure_dir);
            let (clash_threshold, clashes_data) = count_clashes(&contacts);
            let clashes_string: Vec<String> =
                clashes_data.iter().map(|&num| num.to_string()).collect();
            final_report.push(clashes_string);
            println!("Models with more than {clash_threshold} clashes won't be investigated");

            let scores = score_interface(&file_names, &structure_dir, "pTM");
            let scores_string: Vec<String> = scores.iter().map(|&num| num.to_string()).collect();
            final_report.push(scores_string);
            println!("{:?}", file_names);
            println!("{:?}", scores);
        }
        Commands::Iplddt {
            aggregate_1,
            aggregate_2,
            threshold,
        } => {
            let threshold = threshold
                .parse::<f64>()
                .expect("Threshold cannot be cast to f64");
            // Compute plddt at interface for all models
            let all_iplddt = all_iplddt(
                &structure_dir,
                &file_names,
                &aggregate_1,
                &aggregate_2,
                threshold,
            );
            let iplddt_string: Vec<String> =
                all_iplddt.iter().map(|&num| num.to_string()).collect();
            // Register these values in the table
            report_colnames.push(String::from("i-plddt"));
            final_report.push(iplddt_string);
        }
        Commands::Distances {
            filename,
            chain_pairs,
        } => {
            let distances = all_min_distances(&structure_dir, &file_names);
            let distances: Vec<Vec<ChainDistance>> = distances
                .iter()
                .map(|num| filter_chain_pairs(&num, &chain_pairs))
                .collect();

            //let to_keep = filter_chain_pairs(distances, chain_pairs);
            //let distances = _minimal_chain_distances(&filename);
            //let distances = filter_chain_pairs(&distances, &chain_pairs);
            let (pairs, min_distances) = sanitize_data(&file_names, &distances);
            for i in 0..pairs.len() {
                report_colnames.push(String::from(pairs[i].clone()));
                let distances_to_register: Vec<String> = min_distances[i]
                    .iter()
                    .map(|&num| num.to_string())
                    .collect();
                final_report.push(distances_to_register);
            }
        }
        Commands::Scoring {} => {
            let scores = all_scores_computation(&structure_dir, &file_names);
        }
    }
    report_colnames.push(String::from("Models"));
    final_report.push(file_names);
    if let Err(err) = report_to_csv_structured(&output_csv, final_report, report_colnames) {
        eprintln!("Failed to write structured CSV {}: {}", output_csv, err);
    }
}
