use clap::{Parser, Subcommand};
use indexmap::{IndexMap, IndexSet};
use pdbtbx::open;
use polars::prelude::{
    CsvReader,
    CsvWriter,
    DataFrame,
    DataType,
    NamedFrom,
    SerReader,
    SerWriter,
    Series,
};
use std::{
    error::Error,
    fs::File,
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
};

use crate::{
    all_alignment,
    all_contacts,
    all_distances,
    all_iplddt,
    all_min_distances,
    all_scores_computation,
    count_clashes,
    filter_chain_pairs,
    gather_runs,
    parallel_all_alignment,
    sanitize_data,
    score_interface,
    structure_files_from_directory,
    ChainDistance,
    Contact,
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

fn series_to_strings(series: &Series, row_count: usize) -> Result<Vec<String>, Box<dyn Error>> {
    let utf8_series = if matches!(series.dtype(), DataType::String) {
        series.clone()
    } else {
        series.cast(&DataType::String)?
    };
    let chunked = utf8_series.str()?;
    let mut values = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let value = chunked.get(row_idx).unwrap_or("");
        values.push(value.to_string());
    }
    Ok(values)
}

fn load_structured_rows(path: &str) -> Result<(StructuredRows, IndexSet<String>), Box<dyn Error>> {
    let mut rows = StructuredRows::new();
    let mut header_order = IndexSet::new();
    if !Path::new(path).exists() {
        return Ok((rows, header_order));
    }
    let df = CsvReader::from_path(path)?.has_header(true).finish()?;
    let headers: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();
    for header in headers.iter() {
        header_order.insert(header.clone());
    }
    let models_idx = headers.iter().position(|h| h == "Models").unwrap_or(0);
    let row_count = df.height();
    let mut columns: Vec<Vec<String>> = Vec::with_capacity(headers.len());
    for header in headers.iter() {
        let series = df.column(header)?;
        let values = series_to_strings(series, row_count)?;
        columns.push(values);
    }
    for row_idx in 0..row_count {
        let model_value = columns
            .get(models_idx)
            .and_then(|col| col.get(row_idx))
            .cloned()
            .unwrap_or_default();
        if model_value.is_empty() {
            continue;
        }
        let entry = rows
            .entry(model_value.clone())
            .or_insert_with(IndexMap::new);
        for (idx, header) in headers.iter().enumerate() {
            let value = columns
                .get(idx)
                .and_then(|col| col.get(row_idx))
                .cloned()
                .unwrap_or_default();
            entry.insert(header.clone(), value);
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
    let row_count = rows.len();
    let mut column_values: Vec<Vec<String>> = headers
        .iter()
        .map(|_| Vec::with_capacity(row_count))
        .collect();
    for (model, metrics) in rows.iter() {
        for (idx, header) in headers.iter().enumerate() {
            let value = if header == "Models" {
                model.clone()
            } else {
                metrics.get(header).cloned().unwrap_or_default()
            };
            if let Some(column) = column_values.get_mut(idx) {
                column.push(value);
            }
        }
    }
    let series: Vec<Series> = headers
        .iter()
        .zip(column_values)
        .map(|(header, values)| Series::new(header.as_str(), values))
        .collect();
    let mut df = DataFrame::new(series)?;
    let mut file = File::create(path)?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut df)?;
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

#[derive(clap::Args)]
struct CommonArgs {
    structure_dir: String,
    output_csv: String,
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
        chain_ids: String,
        #[arg(value_parser = ["TM-score", "rmsd-cur"], default_value = "TM-score")]
        metric: String,
        rmsd_chains: Option<String>,
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Complete analysis on contacts.
    Contacts {
        /// Path to store the aligned structures
        output_dir: String,
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Compute the plddt at the interface between two group of chains.
    Iplddt {
        /// Aggregated identifiers of the interface's first group of chains (e.g. "AB" or "C")
        aggregate_1: String,
        /// Aggregated identifiers of the interface's second group of chains (e.g. "AB" or "C")
        aggregate_2: String,
        /// Distance threshold between two residues to count as interface
        threshold: String,
        #[command(flatten)]
        common: CommonArgs,
    },
    Distances {
        filename: String,
        chain_pairs: String,
        #[command(flatten)]
        common: CommonArgs,
    },
    Scoring {
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Gather ranked AlphaFold runs into a consolidated output directory.
    Gather {
        /// Path to the runs you want to gather
        #[arg(long)]
        runs_path: String,
        /// List of run names or subdirectories to ignore
        #[arg(long, num_args = 1..)]
        ignore: Vec<String>,
        /// Path to the output of the runs gathering (default: <RUNS_PATH>/all_pdbs)
        #[arg(long)]
        output_path: Option<String>,
        /// Skips the run gathering, only output a csv ranking file
        #[arg(long, default_value_t = false)]
        only_ranking: bool,
        /// If specified, include the .pkl pickle files in the gathered results
        #[arg(long, default_value_t = false)]
        include_pickles: bool,
        /// Include a ranking in the name of the ranked files and mapping CSV
        #[arg(long, default_value_t = false)]
        include_rank: bool,
    },
}

#[derive(Parser)]
#[command(version = "1.0", author = "Nessim Raouraoua")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    /// Compute alignments on single thread instead of multithreaded computations
    #[arg(short, long, default_value_t = false)]
    disable_parallel: bool,
}

pub fn run_from_env() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();
    run(args)
}

pub fn run_from_args<I, T>(args: I) -> Result<(), Box<dyn Error>>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let args = Cli::parse_from(args);
    run(args)
}

fn run(args: Cli) -> Result<(), Box<dyn Error>> {
    let do_in_parallel = !args.disable_parallel;

    let mut report_colnames: Vec<String> = Vec::new();
    let mut final_report: Vec<Vec<String>> = Vec::new();

    match args.command {
        Commands::Gather {
            runs_path,
            ignore,
            output_path,
            only_ranking,
            include_pickles,
            include_rank,
        } => {
            let runs_path = PathBuf::from(runs_path);
            let output_path = output_path
                .map(PathBuf::from)
                .unwrap_or_else(|| runs_path.join("all_pdbs"));
            gather_runs(
                &runs_path,
                &ignore,
                &output_path,
                only_ranking,
                include_pickles,
                include_rank,
            )?;
        }
        Commands::Fit {
            output_dir,
            reference_structure,
            chain_ids,
            metric,
            rmsd_chains,
            common,
        } => {
            let structure_dir = common.structure_dir;
            let file_names = structure_files_from_directory(&structure_dir)?;
            let output_csv = common.output_csv;
            let reference_chain = chain_ids;
            let (pdb1, _errors) = open(&reference_structure)
                .expect(&format!("Failed to open first PDB {}", &reference_structure));
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
            let compute_distance = true;
            if compute_distance {
                let dtype = &metric;
                let distances = all_distances(
                    &reference_structure,
                    &file_names,
                    &output_dir,
                    dtype,
                    &rmsd_chains,
                );
                let distances_string: Vec<String> =
                    distances.iter().map(|&num| num.to_string()).collect();
                let chain_names = match rmsd_chains {
                    Some(chain_group) => format!(" ({})", &chain_group),
                    None => String::from(""),
                };
                let colname = format!(
                    "{} to {}{}",
                    metric,
                    Path::new(&reference_structure)
                        .file_stem()
                        .expect("No basename for this path")
                        .to_string_lossy(),
                    chain_names
                );
                report_colnames.push(colname);
                final_report.push(distances_string);
            }

            report_colnames.push(String::from("Models"));
            final_report.push(file_names);
            if let Err(err) = report_to_csv_structured(&output_csv, final_report, report_colnames) {
                eprintln!("Failed to write structured CSV {}: {}", output_csv, err);
            }
        }
        Commands::Contacts { output_dir: _, common, } => {
            let structure_dir = common.structure_dir;
            let file_names = structure_files_from_directory(&structure_dir)?;
            let output_csv = common.output_csv;

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

            report_colnames.push(String::from("Models"));
            final_report.push(file_names);
            if let Err(err) = report_to_csv_structured(&output_csv, final_report, report_colnames) {
                eprintln!("Failed to write structured CSV {}: {}", output_csv, err);
            }
        }
        Commands::Iplddt {
            aggregate_1,
            aggregate_2,
            threshold,
            common,
        } => {
            let structure_dir = common.structure_dir;
            let file_names = structure_files_from_directory(&structure_dir)?;
            let output_csv = common.output_csv;

            let threshold = threshold
                .parse::<f64>()
                .expect("Threshold cannot be cast to f64");
            let all_iplddt = all_iplddt(
                &structure_dir,
                &file_names,
                &aggregate_1,
                &aggregate_2,
                threshold,
            );
            let iplddt_string: Vec<String> =
                all_iplddt.iter().map(|&num| num.to_string()).collect();
            report_colnames.push(String::from("i-plddt"));
            final_report.push(iplddt_string);

            report_colnames.push(String::from("Models"));
            final_report.push(file_names);
            if let Err(err) = report_to_csv_structured(&output_csv, final_report, report_colnames) {
                eprintln!("Failed to write structured CSV {}: {}", output_csv, err);
            }
        }
        Commands::Distances {
            filename: _,
            chain_pairs,
            common,
        } => {
            let structure_dir = common.structure_dir;
            let file_names = structure_files_from_directory(&structure_dir)?;
            let output_csv = common.output_csv;

            let distances = all_min_distances(&structure_dir, &file_names);
            let distances: Vec<Vec<ChainDistance>> = distances
                .iter()
                .map(|num| filter_chain_pairs(&num, &chain_pairs))
                .collect();

            let (pairs, min_distances) = sanitize_data(&file_names, &distances);
            for i in 0..pairs.len() {
                report_colnames.push(String::from(pairs[i].clone()));
                let distances_to_register: Vec<String> = min_distances[i]
                    .iter()
                    .map(|&num| num.to_string())
                    .collect();
                final_report.push(distances_to_register);
            }
            report_colnames.push(String::from("Models"));
            final_report.push(file_names);
            if let Err(err) = report_to_csv_structured(&output_csv, final_report, report_colnames) {
                eprintln!("Failed to write structured CSV {}: {}", output_csv, err);
            }
        }
        Commands::Scoring { common, } => {
            let structure_dir = common.structure_dir;
            let file_names = structure_files_from_directory(&structure_dir)?;
            let output_csv = common.output_csv;

            let _scores = all_scores_computation(&structure_dir, &file_names);
        }
    }
    Ok(())
}
