use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use nalgebra::Vector3;
use pdbtbx::{save, PDB, ReadOptions, StrictnessLevel};
use rayon::prelude::*;

use crate::alignment::{
    apply_transform, compute_centroid, sanitize_structure, structure_files_from_directory,
    try_collect_atom_positions_ref, try_compute_alignment_transform,
};
use crate::cli::{
    columns_to_structured_rows, load_structured_rows, structured_csv_path, write_structured_csv,
};
use crate::progress::default_progress_style;

type ClusterResult<T> = Result<T, Box<dyn Error + Send + Sync>>;
type ClusterIndices = Vec<Vec<usize>>;

#[derive(Clone, Debug, PartialEq)]
pub struct ReducedPoint {
    pub model: String,
    pub point: Vector3<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterAssignment {
    pub model: String,
    pub point: Vector3<f64>,
    pub cluster_id: usize,
}

fn push_cluster_report_columns(
    assignments: &[ClusterAssignment],
    report_colnames: &mut Vec<String>,
    final_report: &mut Vec<Vec<String>>,
) {
    report_colnames.push(String::from("point_x"));
    final_report.push(
        assignments
            .iter()
            .map(|assignment| assignment.point.x.to_string())
            .collect(),
    );
    report_colnames.push(String::from("point_y"));
    final_report.push(
        assignments
            .iter()
            .map(|assignment| assignment.point.y.to_string())
            .collect(),
    );
    report_colnames.push(String::from("point_z"));
    final_report.push(
        assignments
            .iter()
            .map(|assignment| assignment.point.z.to_string())
            .collect(),
    );
    report_colnames.push(String::from("cluster_id"));
    final_report.push(
        assignments
            .iter()
            .map(|assignment| assignment.cluster_id.to_string())
            .collect(),
    );
    report_colnames.push(String::from("Models"));
    final_report.push(
        assignments
            .iter()
            .map(|assignment| assignment.model.clone())
            .collect(),
    );
}

fn push_reduced_point_report_columns(
    reduced_points: &[ReducedPoint],
    report_colnames: &mut Vec<String>,
    final_report: &mut Vec<Vec<String>>,
) {
    report_colnames.push(String::from("point_x"));
    final_report.push(
        reduced_points
            .iter()
            .map(|point| point.point.x.to_string())
            .collect(),
    );
    report_colnames.push(String::from("point_y"));
    final_report.push(
        reduced_points
            .iter()
            .map(|point| point.point.y.to_string())
            .collect(),
    );
    report_colnames.push(String::from("point_z"));
    final_report.push(
        reduced_points
            .iter()
            .map(|point| point.point.z.to_string())
            .collect(),
    );
    report_colnames.push(String::from("Models"));
    final_report.push(
        reduced_points
            .iter()
            .map(|point| point.model.clone())
            .collect(),
    );
}

fn overwrite_structured_csv(
    csv_filename: &str,
    table: Vec<Vec<String>>,
    headers: Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let structured_path = structured_csv_path(csv_filename);
    let (rows, mut header_order) = columns_to_structured_rows(&headers, &table)?;
    if !header_order.contains("Models") {
        header_order.insert(String::from("Models"));
    }
    write_structured_csv(&structured_path, &rows, &header_order)?;
    Ok(())
}

fn load_cached_reduced_points(
    csv_filename: &str,
    file_names: &[String],
) -> Result<Option<Vec<ReducedPoint>>, Box<dyn Error>> {
    let structured_path = structured_csv_path(csv_filename);
    let (rows, _) = load_structured_rows(&structured_path)?;
    if rows.is_empty() {
        return Ok(None);
    }

    let mut reduced_points = Vec::with_capacity(file_names.len());
    for model_name in file_names {
        let Some(metrics) = rows.get(model_name) else {
            return Ok(None);
        };
        let (Some(x), Some(y), Some(z)) = (
            metrics.get("point_x"),
            metrics.get("point_y"),
            metrics.get("point_z"),
        ) else {
            return Ok(None);
        };

        let x = match x.parse::<f64>() {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        let y = match y.parse::<f64>() {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        let z = match z.parse::<f64>() {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };

        reduced_points.push(ReducedPoint {
            model: model_name.clone(),
            point: Vector3::new(x, y, z),
        });
    }

    Ok(Some(reduced_points))
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    Box::new(IoError::new(ErrorKind::InvalidData, message.into()))
}

fn load_structure(path: &str) -> ClusterResult<PDB> {
    let (mut pdb, _errors) = ReadOptions::default()
        .set_level(StrictnessLevel::Loose)
        .read(path)
        .map_err(|errors| {
            invalid_data(format!(
                "failed to read structure '{path}': {}",
                errors
                    .iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<String>>()
                    .join("; ")
            ))
        })?;
    sanitize_structure(&mut pdb);
    Ok(pdb)
}

fn save_structure(pdb: &PDB, output_path: &str) -> ClusterResult<()> {
    save(pdb, output_path, StrictnessLevel::Loose)
        .map_err(|errors| {
            invalid_data(format!(
                "failed to save structure '{output_path}': {}",
                errors
                    .iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<String>>()
                    .join("; ")
            ))
        })?;
    Ok(())
}

fn ensure_output_dir(output_dir: &str) -> ClusterResult<()> {
    fs::create_dir_all(output_dir).map_err(|err| {
        invalid_data(format!(
            "failed to create aligned structure directory '{output_dir}': {err}"
        ))
    })?;
    Ok(())
}

fn write_aligned_reference(reference: &PDB, reference_path: &str, output_dir: &str) -> ClusterResult<()> {
    let reference_name = Path::new(reference_path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid_data(format!("invalid reference path '{reference_path}'")))?;
    let output_path = format!("{output_dir}/{reference_name}");
    save_structure(reference, &output_path)
}

pub fn compute_reduced_point(
    reference: &PDB,
    model: &mut PDB,
    anchor_chains: &str,
    reduction_chains: &str,
) -> ClusterResult<Vector3<f64>> {
    let (rotation, translation) =
        try_compute_alignment_transform(reference, model, anchor_chains).map_err(invalid_data)?;
    apply_transform(model, rotation, translation);
    let selected_atoms =
        try_collect_atom_positions_ref(model, reduction_chains).map_err(invalid_data)?;
    Ok(compute_centroid(&selected_atoms))
}

pub fn complete_linkage_clusters(points: &[ReducedPoint], cutoff: f64) -> Vec<Vec<usize>> {
    let n_points = points.len();
    if n_points <= 1 {
        return (0..n_points).map(|index| vec![index]).collect();
    }

    let cutoff_sq = cutoff * cutoff;
    let mut distances = vec![vec![0.0; n_points]; n_points];
    for left in 0..n_points {
        for right in (left + 1)..n_points {
            let diff = points[left].point - points[right].point;
            let distance_sq = diff.dot(&diff);
            distances[left][right] = distance_sq;
            distances[right][left] = distance_sq;
        }
    }

    let mut clusters: Vec<Vec<usize>> = (0..n_points).map(|index| vec![index]).collect();
    let mut active = vec![true; n_points];
    let progress = ProgressBar::new(n_points.saturating_sub(1) as u64);
    progress.set_style(default_progress_style());
    progress.set_message("cluster merges");

    loop {
        let mut best_pair: Option<(usize, usize, f64)> = None;

        for left in 0..n_points {
            if !active[left] {
                continue;
            }
            for right in (left + 1)..n_points {
                if !active[right] {
                    continue;
                }
                let distance_sq = distances[left][right];
                if distance_sq <= cutoff_sq {
                    match best_pair {
                        Some((_, _, best_distance_sq)) if distance_sq >= best_distance_sq => {}
                        _ => best_pair = Some((left, right, distance_sq)),
                    }
                }
            }
        }

        let Some((left, right, _)) = best_pair else {
            break;
        };

        for other in 0..n_points {
            if other == left || other == right || !active[other] {
                continue;
            }
            let updated_distance = distances[left][other].max(distances[right][other]);
            distances[left][other] = updated_distance;
            distances[other][left] = updated_distance;
        }

        let mut merged = std::mem::take(&mut clusters[right]);
        clusters[left].append(&mut merged);
        clusters[left].sort_unstable();
        active[right] = false;
        progress.inc(1);
    }

    progress.finish_with_message("cluster merges complete");
    let mut final_clusters: Vec<Vec<usize>> = clusters
        .into_iter()
        .enumerate()
        .filter_map(|(index, cluster)| active[index].then_some(cluster))
        .collect();
    final_clusters.sort_by_key(|cluster| cluster[0]);
    final_clusters
}

fn build_assignments(points: Vec<ReducedPoint>, clusters: ClusterIndices) -> Vec<ClusterAssignment> {
    let mut cluster_ids = vec![0; points.len()];

    for (cluster_idx, cluster) in clusters.iter().enumerate() {
        for &point_idx in cluster {
            cluster_ids[point_idx] = cluster_idx + 1;
        }
    }

    points
        .into_iter()
        .enumerate()
        .map(|(index, point)| ClusterAssignment {
            model: point.model,
            point: point.point,
            cluster_id: cluster_ids[index],
        })
        .collect()
}

pub fn assign_clusters(points: Vec<ReducedPoint>, cutoff: f64) -> Vec<ClusterAssignment> {
    let clusters = complete_linkage_clusters(&points, cutoff);
    build_assignments(points, clusters)
}

fn validate_cutoff(cutoff: f64) -> ClusterResult<()> {
    if cutoff < 0.0 {
        return Err(invalid_data("clustering cutoff must be non-negative"));
    }
    Ok(())
}

pub fn compute_reduced_points(
    reference_path: &str,
    input_dir: &str,
    file_names: &[String],
    anchor_chains: &str,
    reduction_chains: &str,
    parallel: bool,
    aligned_output_dir: Option<&str>,
) -> ClusterResult<Vec<ReducedPoint>> {
    let style = default_progress_style();
    let reference = load_structure(reference_path)?;
    if let Some(output_dir) = aligned_output_dir {
        ensure_output_dir(output_dir)?;
        write_aligned_reference(&reference, reference_path, output_dir)?;
    }

    let compute_point = |model_name: &String| -> ClusterResult<ReducedPoint> {
        let model_path = format!("{input_dir}/{model_name}");
        let mut model = load_structure(&model_path)?;
        let point = compute_reduced_point(&reference, &mut model, anchor_chains, reduction_chains)?;
        if let Some(output_dir) = aligned_output_dir {
            let output_path = format!("{output_dir}/{model_name}");
            save_structure(&model, &output_path)?;
        }
        Ok(ReducedPoint {
            model: model_name.clone(),
            point,
        })
    };

    if parallel {
        file_names
            .par_iter()
            .progress_with_style(style)
            .map(compute_point)
            .collect::<Result<Vec<_>, _>>()
    } else {
        file_names
            .iter()
            .progress_with_style(style)
            .map(compute_point)
            .collect::<Result<Vec<_>, _>>()
    }
}

pub fn cluster_structure_files(
    reference_path: &str,
    input_dir: &str,
    anchor_chains: &str,
    reduction_chains: &str,
    cutoff: f64,
    parallel: bool,
    aligned_output_dir: Option<&str>,
) -> ClusterResult<Vec<ClusterAssignment>> {
    validate_cutoff(cutoff)?;

    let file_names = structure_files_from_directory(input_dir)?;
    cluster_structures(
        reference_path,
        input_dir,
        &file_names,
        anchor_chains,
        reduction_chains,
        cutoff,
        parallel,
        aligned_output_dir,
    )
}

pub fn cluster_structures(
    reference_path: &str,
    input_dir: &str,
    file_names: &[String],
    anchor_chains: &str,
    reduction_chains: &str,
    cutoff: f64,
    parallel: bool,
    aligned_output_dir: Option<&str>,
) -> ClusterResult<Vec<ClusterAssignment>> {
    validate_cutoff(cutoff)?;

    println!(
        "Computing reduced-space clustering on {} structures...",
        file_names.len()
    );
    let total_start = Instant::now();
    let reduction_start = Instant::now();
    let reduced_points = compute_reduced_points(
        reference_path,
        input_dir,
        file_names,
        anchor_chains,
        reduction_chains,
        parallel,
        aligned_output_dir,
    )?;
    println!("Reduced points computed in {:?}", reduction_start.elapsed());

    let clustering_start = Instant::now();
    let assignments = assign_clusters(reduced_points, cutoff);
    println!("Clustering completed in {:?}", clustering_start.elapsed());
    println!("Total clustering workflow took {:?}\n", total_start.elapsed());
    Ok(assignments)
}

pub(crate) fn run_cluster_workflow(
    reference_structure: &str,
    structure_dir: &str,
    anchor_chains: &str,
    reduction_chains: &str,
    cutoff: f64,
    do_in_parallel: bool,
    aligned_output_dir: Option<&str>,
    output_csv: &str,
) -> Result<(), Box<dyn Error>> {
    let file_names = structure_files_from_directory(structure_dir)?;
    let total_start = Instant::now();
    let reduced_points = if aligned_output_dir.is_none() {
        match load_cached_reduced_points(output_csv, &file_names)? {
            Some(points) => {
                println!(
                    "Reusing cached reduced coordinates from {}",
                    structured_csv_path(output_csv)
                );
                println!("Reduced points loaded from cache");
                points
            }
            None => {
                let reduction_start = Instant::now();
                let points = compute_reduced_points(
                    reference_structure,
                    structure_dir,
                    &file_names,
                    anchor_chains,
                    reduction_chains,
                    do_in_parallel,
                    aligned_output_dir,
                )
                .map_err(|err| -> Box<dyn Error> { err })?;

                let mut point_headers = Vec::new();
                let mut point_table = Vec::new();
                push_reduced_point_report_columns(
                    &points,
                    &mut point_headers,
                    &mut point_table,
                );
                overwrite_structured_csv(output_csv, point_table, point_headers)?;
                println!("Reduced points computed in {:?}", reduction_start.elapsed());
                points
            }
        }
    } else {
        let reduction_start = Instant::now();
        let points = compute_reduced_points(
            reference_structure,
            structure_dir,
            &file_names,
            anchor_chains,
            reduction_chains,
            do_in_parallel,
            aligned_output_dir,
        )
        .map_err(|err| -> Box<dyn Error> { err })?;

        let mut point_headers = Vec::new();
        let mut point_table = Vec::new();
        push_reduced_point_report_columns(&points, &mut point_headers, &mut point_table);
        overwrite_structured_csv(output_csv, point_table, point_headers)?;
        println!("Reduced points computed in {:?}", reduction_start.elapsed());
        points
    };

    let clustering_start = Instant::now();
    let assignments = assign_clusters(reduced_points, cutoff);
    let mut assignment_headers = Vec::new();
    let mut assignment_table = Vec::new();
    push_cluster_report_columns(
        &assignments,
        &mut assignment_headers,
        &mut assignment_table,
    );
    overwrite_structured_csv(output_csv, assignment_table, assignment_headers)?;
    println!("Clustering completed in {:?}", clustering_start.elapsed());
    println!("Total clustering workflow took {:?}\n", total_start.elapsed());
    Ok(())
}
