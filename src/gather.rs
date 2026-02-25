use indexmap::IndexMap;
use polars::prelude::{
    CsvReader, CsvWriter, DataFrame, DataFrameJoinOps, DataType, JoinArgs, JoinType, NamedFrom,
    SerReader, SerWriter, Series, SortMultipleOptions,
};
use rayon::prelude::*;
use serde_json::Value;
use std::{
    collections::HashSet,
    error::Error,
    fs::{self, File},
    io::{Error as IoError, ErrorKind, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use std::env;

const JOIN_COLUMNS: [&str; 3] = ["file", "parameters", "model_name"];

fn set_polars_printing() {
    env::set_var("POLARS_FMT_TABLE_FORMATTING", "NOTHING");
    env::set_var("POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES", "1");
    env::set_var("POLARS_FMT_TABLE_CELL_ALIGNMENT", "RIGHT");
}

struct RunRanking {
    data: DataFrame,
    score_key: String,
}

fn read_json_object(path: &Path) -> Result<IndexMap<String, Value>, Box<dyn Error>> {
    let file = File::open(path)?;
    let value: Value = serde_json::from_reader(file)?;
    let obj = value.as_object().ok_or_else(|| {
        Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Ranking JSON is not an object",
        )) as Box<dyn Error>
    })?;
    Ok(obj
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect())
}

fn prediction_key(param: &str, model: &str) -> String {
    format!("{param}_{model}")
}

fn read_order_from_ranking(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let obj = read_json_object(path)?;
    let order_val = obj.get("order").ok_or_else(|| {
        Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Ranking JSON missing order",
        )) as Box<dyn Error>
    })?;
    let order = order_val
        .as_array()
        .ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::InvalidData,
                "Ranking order is not an array",
            )) as Box<dyn Error>
        })?
        .iter()
        .filter_map(|val| val.as_str().map(|s| s.to_string()))
        .collect();
    Ok(order)
}

fn read_score_map(path: &Path) -> Result<(String, IndexMap<String, f64>), Box<dyn Error>> {
    let obj = read_json_object(path)?;
    let (score_key, scores_val) = obj
        .iter()
        .find(|(key, value)| key.as_str() != "order" && value.is_object())
        .ok_or_else(|| {
        Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Ranking JSON has no metric scores object",
        )) as Box<dyn Error>
    })?;
    let scores_obj = scores_val.as_object().ok_or_else(|| {
        Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Ranking scores are not an object",
        )) as Box<dyn Error>
    })?;
    let mut scores = IndexMap::new();
    for (key, value) in scores_obj.iter() {
        let score = value.as_f64().ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::InvalidData,
                "Ranking score is not a float",
            )) as Box<dyn Error>
        })?;
        scores.insert(key.clone(), score);
    }
    Ok((score_key.clone(), scores))
}

struct RunResolution {
    model_to_actual: IndexMap<String, String>,
}

fn resolve_actual_file(
    dir_entries: &HashSet<String>,
    idx: usize,
    model_name: &str,
) -> Option<String> {
    let candidates = [
        format!("ranked_{idx}_unrelaxed_{model_name}.pdb"),
        format!("ranked_{idx}_relaxed_{model_name}.pdb"),
        format!("unrelaxed_{model_name}.pdb"),
        format!("relaxed_{model_name}.pdb"),
        format!("ranked_{idx}_{model_name}.cif"),
        format!("{model_name}.cif"),
        format!("{model_name}.pdb"),
    ];
    if let Some(found) = candidates.iter().find(|name| dir_entries.contains(*name)) {
        return Some(found.clone());
    }
    let suffixes = [format!("{model_name}.pdb"), format!("{model_name}.cif")];
    dir_entries
        .iter()
        .find(|entry| suffixes.iter().any(|suffix| entry.ends_with(suffix)))
        .cloned()
}

fn build_run_resolution(
    run_path: &Path,
    ordered_names: &[String],
) -> Result<RunResolution, Box<dyn Error>> {
    let mut dir_entries = HashSet::new();
    for entry in fs::read_dir(run_path)? {
        let entry = entry?;
        dir_entries.insert(entry.file_name().to_string_lossy().to_string());
    }

    let mut model_to_actual = IndexMap::new();
    let mut missing = Vec::new();
    for (idx, model_name) in ordered_names.iter().enumerate() {
        match resolve_actual_file(&dir_entries, idx, model_name) {
            Some(actual) => {
                model_to_actual.insert(model_name.clone(), actual);
            }
            None => missing.push(format!("ranked_{idx}_unrelaxed_{model_name}.pdb")),
        }
    }

    if !missing.is_empty() {
        return Err(Box::new(IoError::new(
            ErrorKind::NotFound,
            format!("Predictions not present: {}", missing.join(", ")),
        )));
    }

    Ok(RunResolution { model_to_actual })
}

fn rank_all(
    all_runs_path: &Path,
    all_runs: &[String],
    output_path: &Path,
    ranking_type: &str,
) -> Result<IndexMap<String, RunRanking>, Box<dyn Error>> {
    if let Err(err) = fs::create_dir_all(output_path) {
        if err.kind() != ErrorKind::AlreadyExists {
            return Err(Box::new(err));
        }
    }

    let mut ranked_per_run: IndexMap<String, RunRanking> = IndexMap::new();
    for run in all_runs {
        let run_path = all_runs_path.join(run);
        let ranking_path = run_path.join("ranking_debug.json");
        let run_score_path = run_path.join(format!("ranking_{ranking_type}.json"));
        if !run_score_path.is_file() {
            return Err(Box::new(IoError::new(
                ErrorKind::NotFound,
                format!("Missing {}", run_score_path.display()),
            )));
        }
        let model_names = read_order_from_ranking(&ranking_path)?;
        let (ranking_key_score, scores_map) = read_score_map(&run_score_path)?;
        let mut scores = Vec::with_capacity(model_names.len());
        for model in model_names.iter() {
            let score = scores_map.get(model).cloned().ok_or_else(|| {
                Box::new(IoError::new(
                    ErrorKind::InvalidData,
                    format!("Missing score for model {model}"),
                )) as Box<dyn Error>
            })?;
            scores.push(score);
        }
        let resolution = build_run_resolution(&run_path, &model_names).map_err(|err| {
            println!(
                "/!\\ some predictions are not found in the run {}, needs investigation",
                run_path.display()
            );
            println!("{err}\n");
            err
        })?;
        let mut predictions = Vec::with_capacity(model_names.len());
        for name in model_names.iter() {
            let actual = resolution
                .model_to_actual
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    Box::new(IoError::new(
                        ErrorKind::NotFound,
                        "Prediction file mapping missing",
                    )) as Box<dyn Error>
                })?;
            predictions.push(actual);
        }

        let parameter_set = Path::new(run)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| run.clone());
        let parameters: Vec<String> = vec![parameter_set; model_names.len()];

        let df = DataFrame::new(vec![
            Series::new("file", predictions),
            Series::new(&ranking_key_score, scores),
            Series::new("parameters", parameters),
            Series::new("model_name", model_names),
        ])?;

        ranked_per_run.insert(
            run.clone(),
            RunRanking {
                data: df,
                score_key: ranking_key_score,
            },
        );
    }

    Ok(ranked_per_run)
}


fn check_all_runs(
    all_runs_path: &Path,
    ignored_dirs: &HashSet<String>,
    ranking_type: &str,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut considered_runs = Vec::new();
    for entry in fs::read_dir(all_runs_path)? {
        let entry = entry?;
        let run = entry.file_name().to_string_lossy().to_string();
        if ignored_dirs.contains(&run) {
            continue;
        }
        let ranking_path = all_runs_path.join(&run).join(format!("ranking_{ranking_type}.json"));
        if ranking_path.is_file() {
            match read_json_object(&ranking_path) {
                Ok(_) => considered_runs.push(run),
                Err(_) => {
                    println!("Something went wrong with ranking_debug.json for run {run}");
                }
            }
        }
    }
    let formatted_runs = considered_runs.join(" - ");
    println!(
        "These are the {} following runs gathered in the output:\n{formatted_runs}\n",
        considered_runs.len()
    );
    Ok(considered_runs)
}

fn merge_metric_frames(base: DataFrame, other: &DataFrame) -> Result<DataFrame, Box<dyn Error>> {
    Ok(base.join(
        other,
        JOIN_COLUMNS,
        JOIN_COLUMNS,
        JoinArgs::new(JoinType::Left),
    )?)
}

fn make_missing_series(name: &str, dtype: &DataType, len: usize) -> Series {
    match dtype {
        DataType::Float64 => Series::new(name, vec![f64::NAN; len]),
        DataType::Float32 => Series::new(name, vec![f32::NAN; len]),
        _ => Series::full_null(name, len, dtype),
    }
}

fn align_frames_for_vstack(
    base: &mut DataFrame,
    incoming: &mut DataFrame,
) -> Result<(), Box<dyn Error>> {
    let base_cols: Vec<String> = base
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();
    let incoming_cols: Vec<String> = incoming
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();

    for col in base_cols.iter() {
        if !incoming_cols.contains(col) {
            let dtype = base.column(col)?.dtype().clone();
            incoming.with_column(make_missing_series(col, &dtype, incoming.height()))?;
        }
    }

    let updated_base_cols: Vec<String> = base
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();
    let updated_incoming_cols: Vec<String> = incoming
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();

    for col in updated_incoming_cols.iter() {
        if !updated_base_cols.contains(col) {
            let dtype = incoming.column(col)?.dtype().clone();
            base.with_column(make_missing_series(col, &dtype, base.height()))?;
        }
    }

    let final_order: Vec<String> = base
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();
    *incoming = incoming.select(final_order)?;

    Ok(())
}

fn add_iptm_ptm_column(df: &mut DataFrame) -> Result<(), Box<dyn Error>> {
    let iptm = df.column("iptm")?.f64()?;
    let ptm = df.column("ptm")?.f64()?;
    let mut combined = Vec::with_capacity(df.height());
    for (iptm_val, ptm_val) in iptm.into_iter().zip(ptm.into_iter()) {
        let iptm_val = iptm_val.ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::InvalidData,
                "iptm value is null",
            )) as Box<dyn Error>
        })?;
        let ptm_val = ptm_val.ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::InvalidData,
                "ptm value is null",
            )) as Box<dyn Error>
        })?;
        combined.push(0.8 * iptm_val + 0.2 * ptm_val);
    }
    df.with_column(Series::new("iptm+ptm", combined))?;
    Ok(())
}

fn create_global_ranking(
    runs: &[String],
    runs_path: &Path,
    output_path: &Path,
    ranking_types: &[&str],
) -> Result<DataFrame, Box<dyn Error>> {
    let mut all_metrics_ranking: Vec<IndexMap<String, RunRanking>> = Vec::new();
    for ranking_type in ranking_types {
        match rank_all(runs_path, runs, output_path, ranking_type) {
            Ok(metric_ranking) => all_metrics_ranking.push(metric_ranking),
            Err(err) => {
                if let Some(io_err) = err.downcast_ref::<IoError>() {
                    if io_err.kind() == ErrorKind::NotFound {
                        println!("No ranking for {ranking_type} metric.");
                        continue;
                    }
                }
                return Err(err);
            }
        }
    }

    if all_metrics_ranking.is_empty() {
        return Err(Box::new(IoError::new(
            ErrorKind::NotFound,
            "No ranking metrics available",
        )));
    }

    let run_names: Vec<String> = all_metrics_ranking[0]
        .keys()
        .map(|name| name.to_string())
        .collect();
    for metric in all_metrics_ranking.iter() {
        let names: Vec<String> = metric.keys().map(|name| name.to_string()).collect();
        if names != run_names {
            return Err(Box::new(IoError::new(
                ErrorKind::InvalidData,
                format!("Not the same runs found: {:?}", names),
            )));
        }
    }

    let mut all_runs_ranking: Option<DataFrame> = None;
    let mut score_keys: Vec<String> = Vec::new();

    for run in run_names {
        let mut run_ranking = all_metrics_ranking[0]
            .get(&run)
            .ok_or_else(|| {
                Box::new(IoError::new(
                    ErrorKind::NotFound,
                    "Run ranking missing",
                )) as Box<dyn Error>
            })?
            .data
            .clone();
        let mut run_key_score = all_metrics_ranking[0]
            .get(&run)
            .ok_or_else(|| {
                Box::new(IoError::new(
                    ErrorKind::NotFound,
                    "Run score key missing",
                )) as Box<dyn Error>
            })?
            .score_key
            .clone();

        for metric in all_metrics_ranking.iter().skip(1) {
            let other = metric.get(&run).ok_or_else(|| {
                Box::new(IoError::new(
                    ErrorKind::NotFound,
                    "Run ranking missing for metric",
                )) as Box<dyn Error>
            })?;
            run_ranking = merge_metric_frames(run_ranking, &other.data)?;
        }

        if run_key_score == "ranking_score" {
            run_ranking.rename("ranking_score", "af3_ranking_score")?;
            run_key_score = "af3_ranking_score".to_string();
            if run_ranking
                .get_column_names()
                .iter()
                .any(|name| *name == "iptm")
            {
                add_iptm_ptm_column(&mut run_ranking)?;
            }
        }

        all_runs_ranking = match all_runs_ranking {
            Some(mut df) => {
                align_frames_for_vstack(&mut df, &mut run_ranking)?;
                df.vstack_mut(&run_ranking)?;
                Some(df)
            }
            None => Some(run_ranking),
        };
        score_keys.push(run_key_score);
    }

    let mut all_runs_ranking =
        all_runs_ranking.ok_or_else(|| Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Empty ranking data",
        )))?;

    let unique_scores: HashSet<String> = score_keys.iter().cloned().collect();
    let (common_key_score, ordering_score): (String, Vec<String>) = if unique_scores.len() == 1 {
        let key = unique_scores.iter().next().cloned().unwrap_or_default();
        (key.clone(), vec![key])
    } else if unique_scores.contains("af3_ranking_score") && unique_scores.contains("iptm+ptm") {
        (
            "iptm+ptm".to_string(),
            vec!["iptm+ptm".to_string(), "af3_ranking_score".to_string()],
        )
    } else if unique_scores.contains("af3_ranking_score") && unique_scores.contains("plddts") {
        (
            "ptm".to_string(),
            vec!["ptm".to_string(), "af3_ranking_score".to_string()],
        )
    } else {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidData,
            format!("ranking_debug.json in some runs uses different metrics: {:?}", score_keys),
        )));
    };

    let sort_options = SortMultipleOptions::new()
        .with_order_descendings(vec![true; ordering_score.len()]);
    all_runs_ranking = all_runs_ranking.sort(ordering_score.clone(), sort_options)?;

    let scores = all_runs_ranking.column(&common_key_score)?.f64()?;
    let mut global_rank: Vec<i32> = Vec::with_capacity(all_runs_ranking.height());
    let mut current_rank = 0;
    let mut prev_score: Option<f64> = None;
    for (idx, value) in scores.into_iter().enumerate() {
        let score = value.ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::InvalidData,
                "Ranking score value is null",
            )) as Box<dyn Error>
        })?;
        if prev_score.map_or(true, |prev| prev != score) {
            current_rank = (idx + 1) as i32;
            prev_score = Some(score);
        }
        global_rank.push(current_rank);
    }
    all_runs_ranking.with_column(Series::new("global_rank", global_rank))?;

    let mut columns_order = vec![
        "global_rank".to_string(),
        common_key_score.clone(),
        "parameters".to_string(),
        "file".to_string(),
        "model_name".to_string(),
    ];
    for column in all_runs_ranking.get_column_names() {
        if !columns_order.contains(&column.to_string()) {
            columns_order.push(column.to_string());
        }
    }
    let all_runs_ranking = all_runs_ranking.select(columns_order)?;

    Ok(all_runs_ranking)
}

fn global_rank_to_json(
    ranking: &DataFrame,
    output_path: &Path,
) -> Result<IndexMap<String, String>, Box<dyn Error>> {
    let parameters = ranking.column("parameters")?.str()?;
    let model_names = ranking.column("model_name")?.str()?;
    let mut map_pred_run = IndexMap::new();
    for (param_opt, model_opt) in parameters.into_iter().zip(model_names.into_iter()) {
        let param: &str = param_opt.unwrap_or("");
        let model: &str = model_opt.unwrap_or("");
        map_pred_run.insert(prediction_key(param, model), param.to_string());
    }

    let info_cols = ["global_rank", "parameters", "file", "model_name"];
    for metric in ranking.get_column_names() {
        if info_cols.contains(&metric) {
            continue;
        }
        let mut ranking_type = metric.to_string();
        if metric == "iptm+ptm" || metric == "ranking_score" {
            ranking_type = "debug".to_string();
        }

        let scores = ranking.column(metric)?.f64()?;
        let params = ranking.column("parameters")?.str()?;
        let models = ranking.column("model_name")?.str()?;
        let mut rows: Vec<(String, f64)> = Vec::new();
        for (idx, value) in scores.into_iter().enumerate() {
            if let Some(score) = value {
                let param = params.get(idx).unwrap_or("");
                let model = models.get(idx).unwrap_or("");
                rows.push((prediction_key(param, model), score));
            }
        }
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut scores_map = serde_json::Map::new();
        for (pred, score) in rows.iter() {
            scores_map.insert(pred.clone(), Value::from(*score));
        }
        let mut global_ranking = serde_json::Map::new();
        global_ranking.insert(metric.to_string(), Value::Object(scores_map));
        global_ranking.insert(
            "order".to_string(),
            Value::Array(rows.iter().map(|(key, _)| Value::String(key.clone())).collect()),
        );

        let ranking_path = output_path.join(format!("ranking_{ranking_type}.json"));
        let mut file = File::create(&ranking_path)?;
        file.write_all(serde_json::to_string_pretty(&Value::Object(global_ranking))?.as_bytes())?;
    }

    Ok(map_pred_run)
}

fn merge_iplddt_data(
    mut ranking: DataFrame,
    additional: &DataFrame,
) -> Result<DataFrame, Box<dyn Error>> {
    let files = ranking.column("file")?.str()?;
    let parameters = ranking.column("parameters")?.str()?;
    let model_names = ranking.column("model_name")?.str()?;
    let mut extensions = Vec::with_capacity(ranking.height());
    let mut models = Vec::with_capacity(ranking.height());
    for idx in 0..ranking.height() {
        let file = files.get(idx).unwrap_or("");
        let extension = file.split('.').nth(1).unwrap_or("").to_string();
        let parameter = parameters.get(idx).unwrap_or("");
        let model = model_names.get(idx).unwrap_or("");
        let model_name = if extension.is_empty() {
            prediction_key(parameter, model)
        } else {
            format!("{parameter}_{model}.{extension}")
        };
        extensions.push(extension);
        models.push(model_name);
    }
    ranking.with_column(Series::new("extension", extensions))?;
    ranking.with_column(Series::new("Models", models))?;

    let joined =
        ranking.join(additional, ["Models"], ["Models"], JoinArgs::new(JoinType::Left))?;
    let joined = joined.drop("Models")?.drop("extension")?;
    Ok(joined)
}

struct CopyTask {
    structure_src: PathBuf,
    structure_dst: PathBuf,
    pickle_src: Option<PathBuf>,
    pickle_dst: Option<PathBuf>,
}

fn copy_single_task(task: &CopyTask) -> Result<u64, IoError> {
    let copied_bytes = fs::copy(&task.structure_src, &task.structure_dst)?;
    if let (Some(src), Some(dst)) = (&task.pickle_src, &task.pickle_dst) {
        let _ = fs::copy(src, dst);
    }
    Ok(copied_bytes)
}

fn copy_task_batch(tasks: &[CopyTask], workers: usize) -> Result<u64, IoError> {
    if workers <= 1 || tasks.len() <= 1 {
        return tasks
            .iter()
            .try_fold(0_u64, |acc, task| copy_single_task(task).map(|bytes| acc + bytes));
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|err| IoError::new(ErrorKind::Other, err.to_string()))?;

    pool.install(|| {
        tasks
            .par_iter()
            .map(copy_single_task)
            .try_reduce(|| 0_u64, |a, b| Ok(a + b))
    })
}

fn copy_tasks_adaptive(tasks: &[CopyTask]) -> Result<(), Box<dyn Error>> {
    if tasks.is_empty() {
        return Ok(());
    }

    let hw_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let half_threads_cap = std::cmp::max(1, hw_threads / 2);
    let max_workers = std::cmp::min(half_threads_cap, tasks.len().max(1));
    let mut workers = 1usize;
    let batch_size = 32;
    let mut idx = 0;
    let decision_window_batches = 3usize;
    let mut previous_window_median: Option<f64> = None;
    let mut window_throughputs: Vec<f64> = Vec::with_capacity(decision_window_batches);
    let stagnation_tolerance = 0.15f64;
    let mut consecutive_stagnation_windows = 0usize;
    let mut last_action_was_increase = false;
    let mut total_batch_workers = 0.0f64;
    let mut total_batch_throughput = 0.0f64;
    let mut total_batches = 0usize;

    while idx < tasks.len() {
        let end = std::cmp::min(idx + batch_size, tasks.len());
        let batch = &tasks[idx..end];
        let start = Instant::now();
        let copied_bytes = copy_task_batch(batch, workers)?;
        let elapsed_s = start.elapsed().as_secs_f64().max(1e-6);
        let throughput = copied_bytes as f64 / elapsed_s;
        total_batch_workers += workers as f64;
        total_batch_throughput += throughput;
        total_batches += 1;
        window_throughputs.push(throughput);

        if window_throughputs.len() == decision_window_batches {
            let mut sorted_window = window_throughputs.clone();
            sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let current_window_median = sorted_window[decision_window_batches / 2];
            let previous_workers = workers;

            if let Some(previous) = previous_window_median {
                let lower = previous * (1.0 - stagnation_tolerance);
                let upper = previous * (1.0 + stagnation_tolerance);
                if current_window_median < lower {
                    workers = std::cmp::max(1, workers.saturating_sub(2));
                    consecutive_stagnation_windows = 0;
                    last_action_was_increase = false;
                } else if current_window_median <= upper {
                    if last_action_was_increase {
                        workers = std::cmp::max(1, workers.saturating_sub(1));
                        consecutive_stagnation_windows = 0;
                        last_action_was_increase = false;
                    } else {
                        consecutive_stagnation_windows += 1;
                        if consecutive_stagnation_windows >= 2 {
                            workers = std::cmp::min(max_workers, workers + 1);
                            consecutive_stagnation_windows = 0;
                        }
                    }
                } else {
                    workers = std::cmp::min(max_workers, workers + 2);
                    consecutive_stagnation_windows = 0;
                    last_action_was_increase = workers > previous_workers;
                }
            } else {
                workers = std::cmp::min(max_workers, workers + 2);
                consecutive_stagnation_windows = 0;
                last_action_was_increase = workers > previous_workers;
            }

            previous_window_median = Some(current_window_median);
            window_throughputs.clear();
        }

        idx = end;
    }

    if total_batches > 0 {
        let mean_perf_mib_s = (total_batch_throughput / total_batches as f64) / (1024.0 * 1024.0);
        println!(
            "Speed on avg: {:.2} MiB/s",
            mean_perf_mib_s
        );
    }

    Ok(())
}

fn move_and_rename(
    all_runs_path: &Path,
    run_names: &IndexMap<String, String>,
    output_path: &Path,
    ranking: &DataFrame,
    include_pickles: bool,
    include_rank: bool,
) -> Result<(), Box<dyn Error>> {
    let column_names = ranking.get_column_names();
    if column_names.len() < 2 {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Ranking CSV missing score column",
        )));
    }
    let score_key = column_names[1].to_string();
    let parameters = ranking.column("parameters")?.str()?;
    let files = ranking.column("file")?.str()?;
    let model_names = ranking.column("model_name")?.str()?;
    let global_ranks = ranking.column("global_rank")?.i32()?;
    let scores = ranking.column(&score_key)?.f64()?;

    let mut models: Vec<i32> = Vec::new();
    let mut runs: Vec<String> = Vec::new();
    let mut predictions: Vec<String> = Vec::new();
    let mut mapped_names: Vec<String> = Vec::new();
    let mut score_values: Vec<f64> = Vec::new();
    let mut copy_tasks: Vec<CopyTask> = Vec::with_capacity(ranking.height());

    for idx in 0..ranking.height() {
        let run_name = parameters.get(idx).unwrap_or("");
        let prediction_old_file = files.get(idx).unwrap_or("");
        let model_name = model_names.get(idx).unwrap_or("");
        let global_rank = global_ranks.get(idx).unwrap_or(0);
        let ranking_score = scores.get(idx).unwrap_or(0.0);

        if idx == 0 {
            let features_old_name = all_runs_path.join(run_name).join("features.pkl");
            let features_new_name = output_path.join("features.pkl");
            if let Err(_) = fs::copy(&features_old_name, &features_new_name) {
                println!("features.pkl not found in {run_name} run");
            }
        }

        let old_pdb_path = all_runs_path.join(run_name).join(prediction_old_file);
        let structure_extension = if prediction_old_file.ends_with(".pdb") {
            "pdb"
        } else if prediction_old_file.ends_with(".cif") {
            "cif"
        } else {
            return Err(Box::new(IoError::new(
                ErrorKind::InvalidData,
                format!("Unknown structure extension for {}", prediction_old_file),
            )));
        };
        let mut pdb_file = format!("{run_name}_{model_name}.{structure_extension}");
        if include_rank {
            pdb_file = format!("ranked_{}_{}", global_rank - 1, pdb_file);
            models.push(global_rank);
            runs.push(run_name.to_string());
            predictions.push(Path::new(prediction_old_file).to_string_lossy().to_string());
            score_values.push(ranking_score);
            mapped_names.push(Path::new(&pdb_file).to_string_lossy().to_string());
        }
        let new_pdb_path = output_path.join(&pdb_file);
        let mut pickle_src: Option<PathBuf> = None;
        let mut pickle_dst: Option<PathBuf> = None;

        if include_pickles {
            let prediction_name = format!("{run_name}_{model_name}");
            let pickles_run = run_names.get(&prediction_name).ok_or_else(|| {
                Box::new(IoError::new(
                    ErrorKind::NotFound,
                    "Prediction run mapping missing",
                )) as Box<dyn Error>
            })?;
            let mut pickles_path = all_runs_path.join(pickles_run);
            if pickles_path.join("light_pkl").is_dir() {
                pickles_path = pickles_path.join("light_pkl");
            }
            pickle_src = Some(pickles_path.join(format!("result_{model_name}.pkl")));
            pickle_dst = Some(output_path.join(format!("{prediction_name}.pkl")));
        }

        copy_tasks.push(CopyTask {
            structure_src: old_pdb_path,
            structure_dst: new_pdb_path,
            pickle_src,
            pickle_dst,
        });
    }

    copy_tasks_adaptive(&copy_tasks)?;

    if include_rank {
        let mut mapping = DataFrame::new(vec![
            Series::new("model", models),
            Series::new("run", runs),
            Series::new("prediction", predictions),
            Series::new(&score_key, score_values),
            Series::new("mapped_name", mapped_names),
        ])?;
        let mut file = File::create(output_path.join("map.csv"))?;
        CsvWriter::new(&mut file)
            .include_header(true)
            .finish(&mut mapping)?;
    }
    Ok(())
}

pub fn gather_runs(
    runs_path: &Path,
    ignore_runs: &[String],
    output_path: &Path,
    only_ranking: bool,
    include_pickles: bool,
    include_rank: bool,
) -> Result<(), Box<dyn Error>> {
    set_polars_printing();
    let sequence_name = runs_path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| String::from(""));

    let ranking_types = ["debug", "iptm", "ptm", "plddt"];
    let mut ignored = vec![
        String::from("all_pdbs"),
        String::from("all_runs"),
        String::from("msas"),
        String::from("msas_colabfold"),
        String::from("msas_alphafold3"),
    ];
    for run in ignore_runs {
        ignored.push(run.replace('/', ""));
    }
    ignored.sort();
    ignored.dedup();

    let mut existing_entries = HashSet::new();
    for entry in fs::read_dir(runs_path)? {
        let entry = entry?;
        existing_entries.insert(entry.file_name().to_string_lossy().to_string());
    }
    let ignored_dirs: HashSet<String> = ignored
        .into_iter()
        .filter(|dir| existing_entries.contains(dir))
        .collect();

    if !ignored_dirs.is_empty() {
        let mut sorted: Vec<String> = ignored_dirs.iter().cloned().collect();
        sorted.sort();
        println!("The following directories are ignored:\n{}\n", sorted.join(" - "));
    }

    if output_path.exists() && !only_ranking {
        println!(
            "{} from previous iterations exists, deleting it then repeat.",
            output_path.display()
        );
        fs::remove_dir_all(output_path)?;
    }

    let runs = check_all_runs(runs_path, &ignored_dirs, "debug")?;
    if runs.is_empty() {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidData,
            "There should be at least one run to gather",
        )));
    }
    let mut whole_prediction_ranking =
        create_global_ranking(&runs, runs_path, output_path, &ranking_types)?;

    if whole_prediction_ranking.height() == 0 {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Global ranking is empty",
        )));
    }

    for entry in fs::read_dir(runs_path)? {
        let entry = entry?;
        let csv_path = entry.path();
        let is_csv = csv_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext == "csv")
            .unwrap_or(false);
        let is_ranking_csv = csv_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name == "ranking.csv")
            .unwrap_or(false);
        if !is_csv || is_ranking_csv {
            continue;
        }
        let df = CsvReader::from_path(&csv_path)?
            .has_header(true)
            .finish()?;
        if df
            .get_column_names()
            .iter()
            .any(|name| *name == "i-plddt")
        {
            println!("DataFrame containing i-plddt values: {}", csv_path.display());
            println!("{df:?}");
            whole_prediction_ranking = merge_iplddt_data(whole_prediction_ranking, &df)?;
            break;
        }
    }

    println!("{whole_prediction_ranking:?}");
    let mut ranking_csv = whole_prediction_ranking.clone();
    let mut file = File::create(runs_path.join("ranking.csv"))?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut ranking_csv)?;
    let mut ranking_output = whole_prediction_ranking.clone();
    let mut file = File::create(output_path.join("ranking.csv"))?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut ranking_output)?;

    if !only_ranking {
        let pred_run_map = global_rank_to_json(&whole_prediction_ranking, output_path)?;
        println!("Gathering {sequence_name}'s runs");
        if include_pickles {
            println!("Pickle files are also included in the gathering.");
        }
        move_and_rename(
            runs_path,
            &pred_run_map,
            output_path,
            &whole_prediction_ranking,
            include_pickles,
            include_rank,
        )?;
    } else if output_path.exists() {
        fs::remove_dir_all(output_path)?;
    }

    Ok(())
}
