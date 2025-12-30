use align_rs::*;
use std::path::Path;
//use pdbtbx::PDB;
use criterion::{criterion_group, criterion_main};
//use criterion::BenchmarkId;
use criterion::Criterion;

fn parali_atom() {
    let input_dir = String::from("./benches/test/structures");
    let output_dir = String::from("./benches/test/results");
    let reference_structure = String::from(
        "test/structures/ranked_0_afm_dropout_noSM_woTemplates_model_4_multimer_v1_pred_54.pdb",
    );
    let reference_chain = String::from("B");

    let ignore = String::from(
        Path::new(&reference_structure)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );
    let (pdb1, _errors) = pdbtbx::open(reference_structure).expect("Failed to open first PDB");
    let file_names =
        filenames_from_directory(&input_dir).expect("Error reading directory {dirname}");

    parallel_all_alignment(
        &file_names,
        &pdb1,
        &reference_chain,
        &input_dir,
        &output_dir,
        "per_atom",
    );
}

fn parali_pdb() {
    let input_dir = String::from("./benches/test/structures");
    let output_dir = String::from("./benches/test/results");
    let reference_structure = String::from(
        "test/structures/ranked_0_afm_dropout_noSM_woTemplates_model_4_multimer_v1_pred_54.pdb",
    );
    let reference_chain = String::from("B");

    let ignore = String::from(
        Path::new(&reference_structure)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );
    let (pdb1, _errors) = pdbtbx::open(reference_structure).expect("Failed to open first PDB");
    let file_names =
        filenames_from_directory(&input_dir).expect("Error reading directory {dirname}");

    parallel_all_alignment(
        &file_names,
        &pdb1,
        &reference_chain,
        &input_dir,
        &output_dir,
        "on_pdb",
    );
}

fn parali_parpdb() {
    let input_dir = String::from("./benches/test/structures");
    let output_dir = String::from("./benches/test/results");
    let reference_structure = String::from(
        "test/structures/ranked_0_afm_dropout_noSM_woTemplates_model_4_multimer_v1_pred_54.pdb",
    );
    let reference_chain = String::from("B");

    let ignore = String::from(
        Path::new(&reference_structure)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );
    let (pdb1, _errors) = pdbtbx::open(reference_structure).expect("Failed to open first PDB");
    let file_names =
        filenames_from_directory(&input_dir).expect("Error reading directory {dirname}");

    parallel_all_alignment(
        &file_names,
        &pdb1,
        &reference_chain,
        &input_dir,
        &output_dir,
        "on_pdb_par",
    );
}

fn monali_atom() {
    let input_dir = String::from("./benches/test/structures");
    let output_dir = String::from("./benches/test/results");
    let reference_structure = String::from(
        "test/structures/ranked_0_afm_dropout_noSM_woTemplates_model_4_multimer_v1_pred_54.pdb",
    );
    let reference_chain = String::from("B");

    let ignore = String::from(
        Path::new(&reference_structure)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );
    let (pdb1, _errors) = pdbtbx::open(reference_structure).expect("Failed to open first PDB");
    let file_names =
        filenames_from_directory(&input_dir).expect("Error reading directory {dirname}");

    all_alignment(
        &file_names,
        &pdb1,
        &reference_chain,
        &input_dir,
        &output_dir,
        "per_atom",
    );
}

fn monali_pdb() {
    let input_dir = String::from("./benches/test/structures");
    let output_dir = String::from("./benches/test/results");
    let reference_structure = String::from(
        "test/structures/ranked_0_afm_dropout_noSM_woTemplates_model_4_multimer_v1_pred_54.pdb",
    );
    let reference_chain = String::from("B");

    let ignore = String::from(
        Path::new(&reference_structure)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );
    let (pdb1, _errors) = pdbtbx::open(reference_structure).expect("Failed to open first PDB");
    let file_names =
        filenames_from_directory(&input_dir).expect("Error reading directory {dirname}");

    all_alignment(
        &file_names,
        &pdb1,
        &reference_chain,
        &input_dir,
        &output_dir,
        "on_pdb",
    );
}

fn monali_parpdb() {
    let input_dir = String::from("./benches/test/structures");
    let output_dir = String::from("./benches/test/results");
    let reference_structure = String::from(
        "test/structures/ranked_0_afm_dropout_noSM_woTemplates_model_4_multimer_v1_pred_54.pdb",
    );
    let reference_chain = String::from("B");

    let ignore = String::from(
        Path::new(&reference_structure)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );
    let (pdb1, _errors) = pdbtbx::open(reference_structure).expect("Failed to open first PDB");
    let file_names =
        filenames_from_directory(&input_dir).expect("Error reading directory {dirname}");

    all_alignment(
        &file_names,
        &pdb1,
        &reference_chain,
        &input_dir,
        &output_dir,
        "on_pdb_par",
    );
}

fn benchmark_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignments");
    group.warm_up_time(std::time::Duration::from_secs(10));

    group.bench_function("parali_pdb", |b| b.iter(|| parali_pdb()));
    group.bench_function("parali_parpdb", |b| b.iter(|| parali_parpdb()));
    group.bench_function("monali_atom", |b| b.iter(|| monali_atom()));
    group.bench_function("monali_pdb", |b| b.iter(|| monali_pdb()));
    group.bench_function("monali_parpdb", |b| b.iter(|| monali_parpdb()));
    group.bench_function("parali_atom", |b| b.iter(|| parali_atom()));
    group.finish();
}

fn per_atom() {
    let input_dir = String::from("./benches/test/structures");
    let output_dir = String::from("./benches/test/results");
    let reference_structure = String::from(
        "test/structures/ranked_0_afm_dropout_noSM_woTemplates_model_4_multimer_v1_pred_54.pdb",
    );
    let reference_chain = String::from("B");

    let ignore = String::from(
        Path::new(&reference_structure)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );
    let (pdb1, _errors) = pdbtbx::open(reference_structure).expect("Failed to open first PDB");
    let file_names =
        filenames_from_directory(&input_dir).expect("Error reading directory {dirname}");
    parallel_all_alignment(
        &file_names,
        &pdb1,
        &reference_chain,
        &input_dir,
        &output_dir,
        "per_atom",
    );
}

fn on_pdb() {
    let input_dir = String::from("./benches/test/structures");
    let output_dir = String::from("./benches/test/results");
    let reference_structure = String::from(
        "test/structures/ranked_0_afm_dropout_noSM_woTemplates_model_4_multimer_v1_pred_54.pdb",
    );
    let reference_chain = String::from("B");

    let ignore = String::from(
        Path::new(&reference_structure)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );
    let (pdb1, _errors) = pdbtbx::open(reference_structure).expect("Failed to open first PDB");
    let file_names =
        filenames_from_directory(&input_dir).expect("Error reading directory {dirname}");
    parallel_all_alignment(
        &file_names,
        &pdb1,
        &reference_chain,
        &input_dir,
        &output_dir,
        "on_pdb",
    );
}

fn on_pdb_par() {
    let input_dir = String::from("./benches/test/structures");
    let output_dir = String::from("./benches/test/results");
    let reference_structure = String::from(
        "test/structures/ranked_0_afm_dropout_noSM_woTemplates_model_4_multimer_v1_pred_54.pdb",
    );
    let reference_chain = String::from("B");

    let ignore = String::from(
        Path::new(&reference_structure)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );
    let (pdb1, _errors) = pdbtbx::open(reference_structure).expect("Failed to open first PDB");
    let file_names =
        filenames_from_directory(&input_dir).expect("Error reading directory {dirname}");
    parallel_all_alignment(
        &file_names,
        &pdb1,
        &reference_chain,
        &input_dir,
        &output_dir,
        "on_pdb_par",
    );
}

fn benchmark_transformation(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignments");
    group.warm_up_time(std::time::Duration::from_secs(20));

    group.bench_function("per_atom", |b| b.iter(|| per_atom()));
    group.bench_function("on_pdb", |b| b.iter(|| on_pdb()));
    group.bench_function("on_pdb_par", |b| b.iter(|| on_pdb_par()));

    group.finish();
}

/*
criterion_group!(benches, benchmark_alignment);
criterion_main!(benches);
*/

criterion_group!(benches, benchmark_transformation);
criterion_main!(benches);
