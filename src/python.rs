use pdbtbx::Element;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::alignment::{all_alignment, parallel_all_alignment};
use crate::chain_distances::{all_min_distances, minimal_chain_distances, ChainDistance};
use crate::cli;
use crate::contacts::{all_contacts, count_clashes};
use crate::interface::all_iplddt;
use crate::metrics::all_distances;
use crate::scoring::score_interface;
use crate::structure_clustering::{cluster_structures_with_algorithm, compute_reduced_points};
use crate::structure_files_from_directory;
use crate::{assign_clusters_with_algorithm, ClusterAlgorithm, ClusterAssignment, ReducedPoint};

fn resolve_filenames(
    structure_dir: &str,
    file_names: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    if let Some(names) = file_names {
        return Ok(names);
    }
    structure_files_from_directory(structure_dir).map_err(|err| PyIOError::new_err(err.to_string()))
}

fn validate_distance_mode(distance_mode: &str) -> PyResult<()> {
    match distance_mode {
        "TM-score" | "rmsd-cur" => Ok(()),
        _ => Err(PyValueError::new_err(
            "distance_mode must be 'TM-score' or 'rmsd-cur'",
        )),
    }
}

fn chain_distances_to_tuples(distances: Vec<ChainDistance>) -> Vec<(String, String, f64)> {
    distances
        .into_iter()
        .map(|entry| (entry.chain1, entry.chain2, entry.min_distance))
        .collect()
}

fn reduced_points_to_lists(
    points: Vec<ReducedPoint>,
) -> (Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut models = Vec::with_capacity(points.len());
    let mut x = Vec::with_capacity(points.len());
    let mut y = Vec::with_capacity(points.len());
    let mut z = Vec::with_capacity(points.len());

    for point in points {
        models.push(point.model);
        x.push(point.point.x);
        y.push(point.point.y);
        z.push(point.point.z);
    }

    (models, x, y, z)
}

fn assignments_to_cluster_ids(assignments: Vec<ClusterAssignment>) -> Vec<usize> {
    assignments
        .into_iter()
        .map(|assignment| assignment.cluster_id)
        .collect()
}

fn validate_cutoff(cutoff: f64) -> PyResult<()> {
    if cutoff < 0.0 {
        return Err(PyValueError::new_err(
            "cutoff must be a non-negative floating-point value",
        ));
    }
    Ok(())
}

fn parse_cluster_algorithm(value: &str) -> PyResult<ClusterAlgorithm> {
    match value {
        "auto" => Ok(ClusterAlgorithm::Auto),
        "primitive" => Ok(ClusterAlgorithm::Primitive),
        "high-volume" => Ok(ClusterAlgorithm::HighVolume),
        _ => Err(PyValueError::new_err(
            "algorithm must be 'auto', 'primitive', or 'high-volume'",
        )),
    }
}

fn reduced_points_from_lists(
    models: Vec<String>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
) -> PyResult<Vec<ReducedPoint>> {
    let n = models.len();
    if x.len() != n || y.len() != n || z.len() != n {
        return Err(PyValueError::new_err(
            "models, x, y, and z must have the same length",
        ));
    }

    Ok(models
        .into_iter()
        .zip(x)
        .zip(y)
        .zip(z)
        .map(|(((model, x), y), z)| ReducedPoint {
            model,
            point: nalgebra::Vector3::new(x, y, z),
        })
        .collect())
}

/// Return structure filenames (PDB or CIF) sorted by their numeric index.
#[pyfunction]
#[pyo3(text_signature = "(directory, /)")]
fn structure_files(directory: &str) -> PyResult<Vec<String>> {
    structure_files_from_directory(directory).map_err(|err| PyIOError::new_err(err.to_string()))
}

/// Align all structures to a reference chain and write aligned files to output_dir.
#[pyfunction(signature = (
    structure_dir,
    output_dir,
    reference_structure,
    chain_ids,
    *,
    file_names=None,
    parallel=true,
    transformation_method="per_atom"
))]
fn align(
    structure_dir: &str,
    output_dir: &str,
    reference_structure: &str,
    chain_ids: &str,
    file_names: Option<Vec<String>>,
    parallel: bool,
    transformation_method: &str,
) -> PyResult<()> {
    let filenames = resolve_filenames(structure_dir, file_names)?;
    let (pdb1, _errors) = pdbtbx::open(reference_structure).map_err(|err| {
        let message = err
            .iter()
            .map(|item| item.to_string())
            .collect::<Vec<String>>()
            .join("; ");
        PyIOError::new_err(message)
    })?;
    let mut pdb1 = pdb1;
    pdb1.remove_atoms_by(|atom| atom.element() == Some(&Element::H));
    pdb1.full_sort();
    if parallel {
        parallel_all_alignment(
            &filenames,
            &pdb1,
            chain_ids,
            structure_dir,
            output_dir,
            transformation_method,
        );
    } else {
        all_alignment(
            &filenames,
            &pdb1,
            chain_ids,
            structure_dir,
            output_dir,
            transformation_method,
        );
    }
    Ok(())
}

/// Align structures to a reference chain and compute TM-score or RMSD distances.
/// `distance_chains` optionally restricts the distance computation to a chain group
/// for both metrics.
#[pyfunction(signature = (
    structure_dir,
    output_dir,
    reference_structure,
    chain_ids,
    *,
    metric="TM-score",
    distance_chains=None,
    file_names=None,
    parallel=true,
    transformation_method="per_atom"
))]
fn fit(
    structure_dir: &str,
    output_dir: &str,
    reference_structure: &str,
    chain_ids: &str,
    metric: &str,
    distance_chains: Option<String>,
    file_names: Option<Vec<String>>,
    parallel: bool,
    transformation_method: &str,
) -> PyResult<Vec<f64>> {
    validate_distance_mode(metric)?;
    align(
        structure_dir,
        output_dir,
        reference_structure,
        chain_ids,
        file_names.clone(),
        parallel,
        transformation_method,
    )?;
    let filenames = resolve_filenames(structure_dir, file_names)?;
    Ok(all_distances(
        reference_structure,
        &filenames,
        output_dir,
        metric,
        &distance_chains,
    ))
}

/// Compute TM-score or RMSD distances against a reference without alignment.
/// `distance_chains` optionally restricts the distance computation to a chain group
/// for both metrics.
/// Note: this writes a CSV report in the current working directory.
#[pyfunction(signature = (
    structure_dir,
    reference_structure,
    *,
    distance_mode="TM-score",
    distance_chains=None,
    file_names=None
))]
fn distances(
    structure_dir: &str,
    reference_structure: &str,
    distance_mode: &str,
    distance_chains: Option<String>,
    file_names: Option<Vec<String>>,
) -> PyResult<Vec<f64>> {
    validate_distance_mode(distance_mode)?;
    let filenames = resolve_filenames(structure_dir, file_names)?;
    Ok(all_distances(
        reference_structure,
        &filenames,
        structure_dir,
        distance_mode,
        &distance_chains,
    ))
}

/// Compute interface pLDDT for each structure.
#[pyfunction(signature = (structure_dir, aggregate_1, aggregate_2, threshold, *, file_names=None))]
fn iplddt(
    structure_dir: &str,
    aggregate_1: &str,
    aggregate_2: &str,
    threshold: f64,
    file_names: Option<Vec<String>>,
) -> PyResult<Vec<f64>> {
    let filenames = resolve_filenames(structure_dir, file_names)?;
    Ok(all_iplddt(
        structure_dir,
        &filenames,
        aggregate_1,
        aggregate_2,
        threshold,
    ))
}

/// Count atomic clashes per structure; returns (threshold, counts).
#[pyfunction(signature = (structure_dir, *, file_names=None))]
fn clash_counts(structure_dir: &str, file_names: Option<Vec<String>>) -> PyResult<(f64, Vec<f64>)> {
    let filenames = resolve_filenames(structure_dir, file_names)?;
    let contacts = all_contacts(&filenames, structure_dir);
    Ok(count_clashes(&contacts))
}

/// Align submitted models on a reference and reduce each model to one 3D point.
///
/// Returns four parallel lists `(models, x, y, z)` in the same order as the
/// submitted `file_names` or, when `file_names` is omitted, in the default
/// sorted order returned by `massif.structure_files(structure_dir)`.
///
/// Parameters
/// ----------
/// structure_dir:
///     Directory containing the input structure files.
/// reference_structure:
///     Path to the reference structure used for the alignment step.
/// anchor_chains:
///     Concatenated chain identifiers used to align each model on the reference.
/// reduction_chains:
///     Concatenated chain identifiers whose aligned atoms are averaged into one
///     reduced point per model.
/// file_names:
///     Optional ordered subset of filenames to process.
/// parallel:
///     Whether to compute reduced points in parallel.
/// aligned_output_dir:
///     Optional directory where the aligned reference and aligned models are written.
#[pyfunction(signature = (
    structure_dir,
    reference_structure,
    anchor_chains,
    reduction_chains,
    *,
    file_names=None,
    parallel=true,
    aligned_output_dir=None
))]
fn reduce_to_point(
    structure_dir: &str,
    reference_structure: &str,
    anchor_chains: &str,
    reduction_chains: &str,
    file_names: Option<Vec<String>>,
    parallel: bool,
    aligned_output_dir: Option<String>,
) -> PyResult<(Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let filenames = resolve_filenames(structure_dir, file_names)?;
    let points = compute_reduced_points(
        reference_structure,
        structure_dir,
        &filenames,
        anchor_chains,
        reduction_chains,
        parallel,
        aligned_output_dir.as_deref(),
    )
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok(reduced_points_to_lists(points))
}

/// Cluster precomputed reduced coordinates and return cluster ids in the same order.
///
/// This function expects the four lists previously returned by `reduce_to_point`.
/// The `models`, `x`, `y`, and `z` arrays must have the same length and matching
/// order. The returned cluster ids follow that exact same order.
#[pyfunction(signature = (models, x, y, z, cutoff, *, algorithm="auto"))]
fn cluster_coordinates(
    models: Vec<String>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    cutoff: f64,
    algorithm: &str,
) -> PyResult<Vec<usize>> {
    validate_cutoff(cutoff)?;
    let algorithm = parse_cluster_algorithm(algorithm)?;
    let points = reduced_points_from_lists(models, x, y, z)?;
    Ok(assignments_to_cluster_ids(assign_clusters_with_algorithm(
        points, cutoff, algorithm,
    )))
}

/// Run the full clustering workflow and return cluster ids in model order.
///
/// This is a convenience wrapper around the Rust reduction and clustering steps.
/// The returned cluster ids are ordered exactly like the submitted `file_names`
/// or, if `file_names` is omitted, like `massif.structure_files(structure_dir)`.
#[pyfunction(signature = (
    structure_dir,
    reference_structure,
    anchor_chains,
    reduction_chains,
    cutoff,
    *,
    file_names=None,
    parallel=true,
    aligned_output_dir=None,
    algorithm="auto"
))]
fn cluster_models(
    structure_dir: &str,
    reference_structure: &str,
    anchor_chains: &str,
    reduction_chains: &str,
    cutoff: f64,
    file_names: Option<Vec<String>>,
    parallel: bool,
    aligned_output_dir: Option<String>,
    algorithm: &str,
) -> PyResult<Vec<usize>> {
    validate_cutoff(cutoff)?;
    let algorithm = parse_cluster_algorithm(algorithm)?;
    let filenames = resolve_filenames(structure_dir, file_names)?;
    let assignments = cluster_structures_with_algorithm(
        reference_structure,
        structure_dir,
        &filenames,
        anchor_chains,
        reduction_chains,
        cutoff,
        parallel,
        aligned_output_dir.as_deref(),
        algorithm,
    )
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok(assignments_to_cluster_ids(assignments))
}

/// Compute minimal chain-to-chain distances for one structure.
#[pyfunction]
#[pyo3(text_signature = "(pdb_file, /)")]
fn chain_distances(pdb_file: &str) -> PyResult<Vec<(String, String, f64)>> {
    Ok(chain_distances_to_tuples(minimal_chain_distances(pdb_file)))
}

/// Compute minimal chain-to-chain distances for all structures in a directory.
#[pyfunction(signature = (structure_dir, *, file_names=None))]
fn all_chain_distances(
    structure_dir: &str,
    file_names: Option<Vec<String>>,
) -> PyResult<Vec<Vec<(String, String, f64)>>> {
    let filenames = resolve_filenames(structure_dir, file_names)?;
    let distances = all_min_distances(structure_dir, &filenames);
    Ok(distances
        .into_iter()
        .map(chain_distances_to_tuples)
        .collect())
}

/// Placeholder interface scoring; returns zeros for now.
#[pyfunction(signature = (structure_dir, ptm_type="pTM", *, file_names=None))]
fn interface_scores(
    structure_dir: &str,
    ptm_type: &str,
    file_names: Option<Vec<String>>,
) -> PyResult<Vec<f64>> {
    let filenames = resolve_filenames(structure_dir, file_names)?;
    Ok(score_interface(&filenames, structure_dir, ptm_type))
}

/// Run the Rust CLI using process arguments or a provided list.
#[pyfunction(signature = (args=None, /))]
fn run_cli(args: Option<Vec<String>>) -> PyResult<()> {
    let argv = if let Some(mut argv) = args {
        argv.insert(0, String::from("massif"));
        argv
    } else {
        Python::with_gil(|py| -> PyResult<Vec<String>> {
            let sys = py.import("sys")?;
            let argv: Vec<String> = sys.getattr("argv")?.extract()?;
            Ok(argv)
        })?
    };
    cli::run_from_args(argv).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pymodule]
fn massif(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(structure_files, m)?)?;
    m.add_function(wrap_pyfunction!(align, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(distances, m)?)?;
    m.add_function(wrap_pyfunction!(iplddt, m)?)?;
    m.add_function(wrap_pyfunction!(clash_counts, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_to_point, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_models, m)?)?;
    m.add_function(wrap_pyfunction!(chain_distances, m)?)?;
    m.add_function(wrap_pyfunction!(all_chain_distances, m)?)?;
    m.add_function(wrap_pyfunction!(interface_scores, m)?)?;
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    Ok(())
}
