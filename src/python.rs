use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pdbtbx::Element;

use crate::cli;
use crate::alignment::{all_alignment, parallel_all_alignment};
use crate::chain_distances::{all_min_distances, minimal_chain_distances, ChainDistance};
use crate::contacts::{all_contacts, count_clashes};
use crate::interface::all_iplddt;
use crate::metrics::all_distances;
use crate::scoring::score_interface;
use crate::structure_files_from_directory;

fn resolve_filenames(
    structure_dir: &str,
    file_names: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    if let Some(names) = file_names {
        return Ok(names);
    }
    structure_files_from_directory(structure_dir)
        .map_err(|err| PyIOError::new_err(err.to_string()))
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
#[pyfunction(signature = (
    structure_dir,
    output_dir,
    reference_structure,
    chain_ids,
    *,
    metric="TM-score",
    rmsd_chains=None,
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
    rmsd_chains: Option<String>,
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
        &rmsd_chains,
    ))
}

/// Compute TM-score or RMSD distances against a reference without alignment.
/// Note: this writes a CSV report in the current working directory.
#[pyfunction(signature = (
    structure_dir,
    reference_structure,
    *,
    distance_mode="TM-score",
    rmsd_chains=None,
    file_names=None
))]
fn distances(
    structure_dir: &str,
    reference_structure: &str,
    distance_mode: &str,
    rmsd_chains: Option<String>,
    file_names: Option<Vec<String>>,
) -> PyResult<Vec<f64>> {
    validate_distance_mode(distance_mode)?;
    let filenames = resolve_filenames(structure_dir, file_names)?;
    Ok(all_distances(
        reference_structure,
        &filenames,
        structure_dir,
        distance_mode,
        &rmsd_chains,
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

/// Compute minimal chain-to-chain distances for one structure.
#[pyfunction]
#[pyo3(text_signature = "(pdb_file, /)")]
fn chain_distances(pdb_file: &str) -> PyResult<Vec<(String, String, f64)>> {
    Ok(chain_distances_to_tuples(minimal_chain_distances(
        pdb_file,
    )))
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
    m.add_function(wrap_pyfunction!(chain_distances, m)?)?;
    m.add_function(wrap_pyfunction!(all_chain_distances, m)?)?;
    m.add_function(wrap_pyfunction!(interface_scores, m)?)?;
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    Ok(())
}
