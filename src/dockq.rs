use std::time::Instant;

use indicatif::ParallelProgressIterator;
use pdbtbx::{Element, ReadOptions, StrictnessLevel, PDB};
use rayon::prelude::*;

use crate::alignment::structure_file_path;
use crate::progress::default_progress_style;
use dockq_rs::{
    all_interfaces_contacts, compute_dockq, DockQConfig, DockQError, DockQMode, DockQPartners,
    DockQResult,
};

/// DockQ of one model interface against the native.
#[derive(Clone, Debug)]
pub struct ModelInterfaceDockQ {
    /// Model file name.
    pub model: String,
    /// Scored receptor/ligand interface.
    pub partners: DockQPartners,
    /// Score or failure for this model interface.
    pub result: Result<DockQResult, DockQError>,
}

/// A structure read as dockq-rs's file readers do in the standard and CAPRI peptide modes: loose parsing, hydrogens removed, residues sorted.
fn read_dockq_structure(path: &str) -> Result<PDB, DockQError> {
    let mut options = ReadOptions::default();
    options.set_level(StrictnessLevel::Loose);
    let (mut structure, _warnings) =
        options
            .read(path)
            .map_err(|errors| DockQError::StructureRead {
                path: path.to_string(),
                message: errors
                    .iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<String>>()
                    .join("; "),
            })?;
    structure.remove_atoms_by(|atom| atom.element() == Some(&Element::H));
    structure.full_sort();
    Ok(structure)
}

/// The chain couples of the native model
fn native_interfaces(native: &PDB, config: &DockQConfig) -> Vec<DockQPartners> {
    let mut polymer = native.clone();
    polymer.remove_atoms_by(|atom| atom.hetero());
    polymer.remove_empty();
    all_interfaces_contacts(&polymer, config)
        .into_iter()
        .filter(|interface| matches!(&interface.result, Ok(contacts) if !contacts.is_empty()))
        .map(|interface| interface.partners)
        .collect()
}

/// DockQ of a set of models against one native structure, for the given interface or, when `partners` is `None`, for every native chain couple
///
/// Rows order follow `pdb_file_names`, then the interfaces in native chain order.
///
/// # Errors
///
/// Returns [`DockQError`] if the native cannot be read or `config` is in the small-molecule mode, whose mmCIF chains dockq-rs reads
/// differently (see `dockq_rs::compute_dockq_from_files`).
pub fn all_dockq_against_native(
    native_path: &str,
    pdb_file_names: &[String],
    input_dir: &str,
    partners: Option<&DockQPartners>,
    config: &DockQConfig,
) -> Result<Vec<ModelInterfaceDockQ>, DockQError> {
    if matches!(config.mode, DockQMode::SmallMolecule) {
        return Err(DockQError::InvalidMode(String::from(
            "the small-molecule mode is not supported against a set of models",
        )));
    }
    let native = read_dockq_structure(native_path)?;
    let interfaces = match partners {
        Some(partners) => vec![partners.clone()],
        None => native_interfaces(&native, config),
    };

    println!(
        "Computing DockQ of {} structures against {} on {} interface(s)...",
        pdb_file_names.len(),
        native_path,
        interfaces.len()
    );
    let start = Instant::now();
    let style = default_progress_style();

    let rows: Vec<Vec<ModelInterfaceDockQ>> = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|model_name| {
            let model = read_dockq_structure(&structure_file_path(input_dir, model_name));
            interfaces
                .iter()
                .map(|interface| ModelInterfaceDockQ {
                    model: model_name.clone(),
                    partners: interface.clone(),
                    result: match &model {
                        Ok(model) => compute_dockq(&native, model, interface, config),
                        Err(err) => Err(err.clone()),
                    },
                })
                .collect()
        })
        .collect();

    println!("Took {:?}\n", start.elapsed());
    Ok(rows.into_iter().flatten().collect())
}
