use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use nalgebra::Vector3;
use pdbtbx::{save, ReadOptions, StrictnessLevel, PDB};
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};

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
const HIGH_VOLUME_AUTO_THRESHOLD: usize = 2_048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterAlgorithm {
    Auto,
    Primitive,
    HighVolume,
}

impl ClusterAlgorithm {
    pub fn resolve(self, point_count: usize) -> Self {
        match self {
            Self::Auto if point_count >= HIGH_VOLUME_AUTO_THRESHOLD => Self::HighVolume,
            Self::Auto => Self::Primitive,
            algorithm => algorithm,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Primitive => "primitive",
            Self::HighVolume => "high-volume",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ReducedPointWrapper {
    index: usize,
    coordinates: [f64; 3],
}

impl RTreeObject for ReducedPointWrapper {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.coordinates)
    }
}

impl PointDistance for ReducedPointWrapper {
    fn distance_2(&self, point: &[f64; 3]) -> f64 {
        let dx = self.coordinates[0] - point[0];
        let dy = self.coordinates[1] - point[1];
        let dz = self.coordinates[2] - point[2];
        dx * dx + dy * dy + dz * dz
    }

    fn contains_point(&self, point: &[f64; 3]) -> bool {
        self.distance_2(point) == 0.0
    }
}

#[derive(Clone, Copy, Debug)]
struct ClusterEdge {
    distance_sq: f64,
    left: usize,
    right: usize,
}

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
    save(pdb, output_path, StrictnessLevel::Loose).map_err(|errors| {
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

fn write_aligned_reference(
    reference: &PDB,
    reference_path: &str,
    output_dir: &str,
) -> ClusterResult<()> {
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
    complete_linkage_clusters_with_algorithm(points, cutoff, ClusterAlgorithm::Auto)
}

pub fn complete_linkage_clusters_with_algorithm(
    points: &[ReducedPoint],
    cutoff: f64,
    algorithm: ClusterAlgorithm,
) -> Vec<Vec<usize>> {
    match algorithm.resolve(points.len()) {
        ClusterAlgorithm::Auto => unreachable!("auto is always resolved before clustering"),
        ClusterAlgorithm::Primitive => complete_linkage_clusters_primitive(points, cutoff),
        ClusterAlgorithm::HighVolume => complete_linkage_clusters_high_volume(points, cutoff),
    }
}

pub fn complete_linkage_clusters_primitive(
    points: &[ReducedPoint],
    cutoff: f64,
) -> Vec<Vec<usize>> {
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

fn collect_cutoff_edges(points: &[ReducedPoint], cutoff: f64) -> Vec<ClusterEdge> {
    let cutoff_sq = cutoff * cutoff;
    let wrappers: Vec<ReducedPointWrapper> = points
        .iter()
        .enumerate()
        .map(|(index, point)| ReducedPointWrapper {
            index,
            coordinates: [point.point.x, point.point.y, point.point.z],
        })
        .collect();
    let tree = RTree::bulk_load(wrappers.clone());
    let progress = ProgressBar::new(wrappers.len() as u64);
    progress.set_style(default_progress_style());
    progress.set_message("cutoff neighbor search");

    let mut edges = Vec::new();
    for wrapper in &wrappers {
        for neighbor in tree.locate_within_distance(wrapper.coordinates, cutoff_sq) {
            if neighbor.index <= wrapper.index {
                continue;
            }
            let dx = wrapper.coordinates[0] - neighbor.coordinates[0];
            let dy = wrapper.coordinates[1] - neighbor.coordinates[1];
            let dz = wrapper.coordinates[2] - neighbor.coordinates[2];
            let distance_sq = dx * dx + dy * dy + dz * dz;
            if distance_sq <= cutoff_sq {
                edges.push(ClusterEdge {
                    distance_sq,
                    left: wrapper.index,
                    right: neighbor.index,
                });
            }
        }
        progress.inc(1);
    }
    progress.finish_with_message("cutoff neighbor search complete");
    edges.sort_unstable_by(|left, right| {
        left.distance_sq
            .total_cmp(&right.distance_sq)
            .then_with(|| left.left.cmp(&right.left))
            .then_with(|| left.right.cmp(&right.right))
    });
    edges
}

fn find_cluster_root(parent: &mut [usize], index: usize) -> usize {
    if parent[index] == index {
        return index;
    }
    let root = find_cluster_root(parent, parent[index]);
    parent[index] = root;
    root
}

fn pop_next_eligible_pair(
    eligible_pairs: &mut BinaryHeap<Reverse<(usize, usize)>>,
    active: &[bool],
    cluster_sizes: &[usize],
    edge_counts: &[HashMap<usize, usize>],
) -> Option<(usize, usize)> {
    while let Some(Reverse((left, right))) = eligible_pairs.pop() {
        if !active[left] || !active[right] {
            continue;
        }
        let required = cluster_sizes[left] * cluster_sizes[right];
        let Some(&observed) = edge_counts[left].get(&right) else {
            continue;
        };
        if observed == required {
            return Some((left, right));
        }
    }
    None
}

fn register_observed_edge(
    left: usize,
    right: usize,
    cluster_sizes: &[usize],
    edge_counts: &mut [HashMap<usize, usize>],
    eligible_pairs: &mut BinaryHeap<Reverse<(usize, usize)>>,
) {
    let updated = {
        let left_counts = &mut edge_counts[left];
        let entry = left_counts.entry(right).or_insert(0);
        *entry += 1;
        *entry
    };
    {
        let right_counts = &mut edge_counts[right];
        right_counts.insert(left, updated);
    }
    if updated == cluster_sizes[left] * cluster_sizes[right] {
        eligible_pairs.push(Reverse((left, right)));
    }
}

fn merge_high_volume_clusters(
    left: usize,
    right: usize,
    parent: &mut [usize],
    active: &mut [bool],
    cluster_sizes: &mut [usize],
    clusters: &mut [Vec<usize>],
    edge_counts: &mut [HashMap<usize, usize>],
    eligible_pairs: &mut BinaryHeap<Reverse<(usize, usize)>>,
) {
    let merged_size = cluster_sizes[left] + cluster_sizes[right];
    let right_neighbors = std::mem::take(&mut edge_counts[right]);

    parent[right] = left;
    active[right] = false;
    cluster_sizes[left] = merged_size;
    cluster_sizes[right] = 0;
    let (left_clusters, right_clusters) = clusters.split_at_mut(right);
    left_clusters[left].append(&mut right_clusters[0]);

    edge_counts[left].remove(&right);

    for (other, right_count) in right_neighbors {
        if other == left {
            continue;
        }
        {
            let other_counts = &mut edge_counts[other];
            other_counts.remove(&right);
        }
        if !active[other] {
            continue;
        }
        let merged_count = edge_counts[left].get(&other).copied().unwrap_or(0) + right_count;
        {
            let left_counts = &mut edge_counts[left];
            left_counts.insert(other, merged_count);
        }
        {
            let other_counts = &mut edge_counts[other];
            other_counts.insert(left, merged_count);
        }
        if merged_count == cluster_sizes[left] * cluster_sizes[other] {
            let pair = if left < other {
                (left, other)
            } else {
                (other, left)
            };
            eligible_pairs.push(Reverse(pair));
        }
    }
}

pub fn complete_linkage_clusters_high_volume(
    points: &[ReducedPoint],
    cutoff: f64,
) -> Vec<Vec<usize>> {
    let n_points = points.len();
    if n_points <= 1 {
        return (0..n_points).map(|index| vec![index]).collect();
    }

    let edges = collect_cutoff_edges(points, cutoff);
    let mut parent: Vec<usize> = (0..n_points).collect();
    let mut active = vec![true; n_points];
    let mut cluster_sizes = vec![1; n_points];
    let mut clusters: Vec<Vec<usize>> = (0..n_points).map(|index| vec![index]).collect();
    let mut edge_counts: Vec<HashMap<usize, usize>> =
        (0..n_points).map(|_| HashMap::new()).collect();
    let mut eligible_pairs = BinaryHeap::new();

    let progress = ProgressBar::new(edges.len() as u64);
    progress.set_style(default_progress_style());
    progress.set_message("high-volume cluster batches");

    let mut edge_index = 0;
    while edge_index < edges.len() {
        let current_distance_sq = edges[edge_index].distance_sq;
        while edge_index < edges.len()
            && edges[edge_index]
                .distance_sq
                .total_cmp(&current_distance_sq)
                == Ordering::Equal
        {
            let edge = edges[edge_index];
            let left = find_cluster_root(&mut parent, edge.left);
            let right = find_cluster_root(&mut parent, edge.right);
            if left != right {
                let (left, right) = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                register_observed_edge(
                    left,
                    right,
                    &cluster_sizes,
                    &mut edge_counts,
                    &mut eligible_pairs,
                );
            }
            edge_index += 1;
        }

        while let Some((left, right)) =
            pop_next_eligible_pair(&mut eligible_pairs, &active, &cluster_sizes, &edge_counts)
        {
            merge_high_volume_clusters(
                left,
                right,
                &mut parent,
                &mut active,
                &mut cluster_sizes,
                &mut clusters,
                &mut edge_counts,
                &mut eligible_pairs,
            );
        }
        progress.set_position(edge_index as u64);
    }

    progress.finish_with_message("high-volume cluster batches complete");
    let mut final_clusters: Vec<Vec<usize>> = clusters
        .into_iter()
        .enumerate()
        .filter_map(|(index, mut cluster)| {
            if !active[index] {
                return None;
            }
            cluster.sort_unstable();
            Some(cluster)
        })
        .collect();
    final_clusters.sort_by_key(|cluster| cluster[0]);
    final_clusters
}

fn build_assignments(
    points: Vec<ReducedPoint>,
    clusters: ClusterIndices,
) -> Vec<ClusterAssignment> {
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
    assign_clusters_with_algorithm(points, cutoff, ClusterAlgorithm::Auto)
}

pub fn assign_clusters_with_algorithm(
    points: Vec<ReducedPoint>,
    cutoff: f64,
    algorithm: ClusterAlgorithm,
) -> Vec<ClusterAssignment> {
    let clusters = complete_linkage_clusters_with_algorithm(&points, cutoff, algorithm);
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
    cluster_structure_files_with_algorithm(
        reference_path,
        input_dir,
        anchor_chains,
        reduction_chains,
        cutoff,
        parallel,
        aligned_output_dir,
        ClusterAlgorithm::Auto,
    )
}

pub fn cluster_structure_files_with_algorithm(
    reference_path: &str,
    input_dir: &str,
    anchor_chains: &str,
    reduction_chains: &str,
    cutoff: f64,
    parallel: bool,
    aligned_output_dir: Option<&str>,
    cluster_algorithm: ClusterAlgorithm,
) -> ClusterResult<Vec<ClusterAssignment>> {
    validate_cutoff(cutoff)?;

    let file_names = structure_files_from_directory(input_dir)?;
    cluster_structures_with_algorithm(
        reference_path,
        input_dir,
        &file_names,
        anchor_chains,
        reduction_chains,
        cutoff,
        parallel,
        aligned_output_dir,
        cluster_algorithm,
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
    cluster_structures_with_algorithm(
        reference_path,
        input_dir,
        file_names,
        anchor_chains,
        reduction_chains,
        cutoff,
        parallel,
        aligned_output_dir,
        ClusterAlgorithm::Auto,
    )
}

pub fn cluster_structures_with_algorithm(
    reference_path: &str,
    input_dir: &str,
    file_names: &[String],
    anchor_chains: &str,
    reduction_chains: &str,
    cutoff: f64,
    parallel: bool,
    aligned_output_dir: Option<&str>,
    cluster_algorithm: ClusterAlgorithm,
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
    let resolved_algorithm = cluster_algorithm.resolve(reduced_points.len());
    println!(
        "Using {} clustering algorithm on {} reduced points",
        resolved_algorithm.as_str(),
        reduced_points.len()
    );
    let assignments = assign_clusters_with_algorithm(reduced_points, cutoff, resolved_algorithm);
    println!("Clustering completed in {:?}", clustering_start.elapsed());
    println!(
        "Total clustering workflow took {:?}\n",
        total_start.elapsed()
    );
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
    cluster_algorithm: ClusterAlgorithm,
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
                push_reduced_point_report_columns(&points, &mut point_headers, &mut point_table);
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
    let resolved_algorithm = cluster_algorithm.resolve(reduced_points.len());
    println!(
        "Using {} clustering algorithm on {} reduced points",
        resolved_algorithm.as_str(),
        reduced_points.len()
    );
    let assignments = assign_clusters_with_algorithm(reduced_points, cutoff, resolved_algorithm);
    let mut assignment_headers = Vec::new();
    let mut assignment_table = Vec::new();
    push_cluster_report_columns(&assignments, &mut assignment_headers, &mut assignment_table);
    overwrite_structured_csv(output_csv, assignment_table, assignment_headers)?;
    println!("Clustering completed in {:?}", clustering_start.elapsed());
    println!(
        "Total clustering workflow took {:?}\n",
        total_start.elapsed()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        assign_clusters_with_algorithm, ClusterAlgorithm, ReducedPoint, HIGH_VOLUME_AUTO_THRESHOLD,
    };
    use nalgebra::Vector3;
    use std::time::Instant;

    #[derive(Clone, Copy)]
    struct DeterministicRng {
        state: u64,
    }

    impl DeterministicRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }

        fn next_f64(&mut self, max: f64) -> f64 {
            (self.next_u64() % 10_000) as f64 * max / 10_000.0
        }
    }

    fn reduced_points(coords: &[(f64, f64, f64)]) -> Vec<ReducedPoint> {
        coords
            .iter()
            .enumerate()
            .map(|(index, (x, y, z))| ReducedPoint {
                model: format!("model_{index}"),
                point: Vector3::new(*x, *y, *z),
            })
            .collect()
    }

    #[test]
    fn high_volume_matches_primitive_on_tie_heavy_example() {
        let points = reduced_points(&[
            (2.0, 5.0, 3.0),
            (5.0, 1.0, 3.0),
            (4.0, 5.0, 0.0),
            (2.0, 2.0, 1.0),
            (1.0, 2.0, 4.0),
        ]);
        let cutoff = 4.00435051305741_f64;

        let primitive =
            assign_clusters_with_algorithm(points.clone(), cutoff, ClusterAlgorithm::Primitive);
        let high_volume =
            assign_clusters_with_algorithm(points, cutoff, ClusterAlgorithm::HighVolume);

        assert_eq!(primitive, high_volume);
    }

    #[test]
    fn high_volume_matches_primitive_on_randomized_inputs() {
        let mut rng = DeterministicRng::new(42);

        for n_points in 2..40 {
            for _ in 0..250 {
                let coords: Vec<(f64, f64, f64)> = (0..n_points)
                    .map(|_| {
                        (
                            rng.next_f64(6.0).round(),
                            rng.next_f64(6.0).round(),
                            rng.next_f64(6.0).round(),
                        )
                    })
                    .collect();
                let cutoff = 0.5 + rng.next_f64(4.5);
                let points = reduced_points(&coords);
                let primitive = assign_clusters_with_algorithm(
                    points.clone(),
                    cutoff,
                    ClusterAlgorithm::Primitive,
                );
                let high_volume =
                    assign_clusters_with_algorithm(points, cutoff, ClusterAlgorithm::HighVolume);
                assert_eq!(primitive, high_volume, "failed for cutoff {cutoff}");
            }
        }
    }

    #[test]
    fn auto_switches_to_high_volume_for_large_inputs() {
        assert_eq!(
            ClusterAlgorithm::Auto.resolve(HIGH_VOLUME_AUTO_THRESHOLD - 1),
            ClusterAlgorithm::Primitive
        );
        assert_eq!(
            ClusterAlgorithm::Auto.resolve(HIGH_VOLUME_AUTO_THRESHOLD),
            ClusterAlgorithm::HighVolume
        );
    }

    #[test]
    #[ignore]
    fn benchmark_high_volume_against_primitive() {
        let mut points = Vec::new();
        for cluster_idx in 0..700 {
            let base = cluster_idx as f64 * 25.0;
            for offset in 0..4 {
                points.push(ReducedPoint {
                    model: format!("model_{}_{}", cluster_idx, offset),
                    point: Vector3::new(base + offset as f64 * 0.2, offset as f64 * 0.15, 0.0),
                });
            }
        }
        let cutoff = 0.75;

        let primitive_start = Instant::now();
        let primitive =
            assign_clusters_with_algorithm(points.clone(), cutoff, ClusterAlgorithm::Primitive);
        let primitive_elapsed = primitive_start.elapsed();

        let high_volume_start = Instant::now();
        let high_volume =
            assign_clusters_with_algorithm(points, cutoff, ClusterAlgorithm::HighVolume);
        let high_volume_elapsed = high_volume_start.elapsed();

        assert_eq!(primitive, high_volume);
        println!(
            "primitive={:?}, high_volume={:?}",
            primitive_elapsed, high_volume_elapsed
        );
    }
}
