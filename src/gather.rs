use indexmap::IndexMap;
use polars::prelude::{CsvWriter, DataFrame, NamedFrom, SerWriter, Series};
use serde_json::Value;
use std::{
    collections::HashSet,
    env,
    error::Error,
    fs::{self, File},
    io::{Error as IoError, ErrorKind, Write},
    path::Path,
};

type PredictionRunMap = IndexMap<String, String>;

#[derive(Debug)]
struct RankingData {
    metric_key: String,
    scores: IndexMap<String, f64>,
    order: Vec<String>,
}

struct RunResolution {
    model_to_actual: IndexMap<String, String>,
}

fn read_ranking(path: &Path) -> Result<RankingData, Box<dyn Error>> {
    let file = File::open(path)?;
    let value: Value = serde_json::from_reader(file)?;
    let obj = value.as_object().ok_or_else(|| {
        Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Ranking JSON is not an object",
        )) as Box<dyn Error>
    })?;
    let metric_key = obj
        .keys()
        .find(|key| key.as_str() != "order")
        .cloned()
        .ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::InvalidData,
                "Ranking JSON is missing metric key",
            )) as Box<dyn Error>
        })?;
    let scores_val = obj.get(&metric_key).ok_or_else(|| {
        Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Ranking JSON missing metric scores",
        )) as Box<dyn Error>
    })?;
    let scores_obj = scores_val.as_object().ok_or_else(|| {
        Box::new(IoError::new(
            ErrorKind::InvalidData,
            "Ranking metric scores are not an object",
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
    let mut order = Vec::new();
    if let Some(order_val) = obj.get("order").and_then(|val| val.as_array()) {
        for item in order_val {
            if let Some(pred) = item.as_str() {
                order.push(pred.to_string());
            }
        }
    }
    Ok(RankingData {
        metric_key,
        scores,
        order,
    })
}

fn check_all_runs(
    runs_path: &Path,
    ignored_dirs: &HashSet<String>,
    ranking_type: &str,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut considered_runs = Vec::new();
    let mut entries: Vec<String> = Vec::new();
    for entry in fs::read_dir(runs_path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        entries.push(name);
    }
    entries.sort();
    for run in entries {
        if ignored_dirs.contains(&run) {
            continue;
        }
        let ranking_path = runs_path.join(&run).join(format!("ranking_{ranking_type}.json"));
        if ranking_path.is_file() {
            match read_ranking(&ranking_path) {
                Ok(_) => considered_runs.push(run),
                Err(_) => {
                    println!("Something went wrong with ranking_debug.json for run {run}");
                }
            }
        }
    }
    let formatted_runs = considered_runs.join("\n");
    println!(
        "These are the following runs gathered in the output:\n{formatted_runs}\n"
    );
    Ok(considered_runs)
}

fn resolve_actual_file(
    dir_entries: &HashSet<String>,
    idx: usize,
    model_name: &str,
) -> Option<String> {
    let expected = format!("ranked_{idx}_unrelaxed_{model_name}.pdb");
    let expected_relaxed = format!("ranked_{idx}_relaxed_{model_name}.pdb");
    let unranked_unrelaxed = format!("unrelaxed_{model_name}.pdb");
    let unranked_relaxed = format!("relaxed_{model_name}.pdb");
    for candidate in [
        &expected,
        &expected_relaxed,
        &unranked_unrelaxed,
        &unranked_relaxed,
    ] {
        if dir_entries.contains(candidate) {
            return Some(candidate.to_string());
        }
    }
    let suffix = format!("{model_name}.pdb");
    dir_entries
        .iter()
        .find(|entry| entry.ends_with(&suffix))
        .cloned()
}

fn build_run_resolution(
    run_path: &Path,
    ranking: &RankingData,
) -> Result<RunResolution, Box<dyn Error>> {
    let mut dir_entries = HashSet::new();
    for entry in fs::read_dir(run_path)? {
        let entry = entry?;
        dir_entries.insert(entry.file_name().to_string_lossy().to_string());
    }

    let mut model_to_actual = IndexMap::new();
    let model_names: Vec<String> = ranking.scores.keys().cloned().collect();
    let mut missing = Vec::new();
    for (idx, model_name) in model_names.iter().enumerate() {
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
    runs_path: &Path,
    runs: &[String],
    output_path: &Path,
    ranking_type: &str,
) -> Result<(String, IndexMap<String, RunResolution>), Box<dyn Error>> {
    if let Err(err) = fs::create_dir_all(output_path) {
        if err.kind() != ErrorKind::AlreadyExists {
            return Err(Box::new(err));
        }
    }

    let mut ranking_key = String::new();
    let mut rows: Vec<(String, String, f64)> = Vec::new();
    let mut resolutions = IndexMap::new();

    for run in runs {
        let run_path = runs_path.join(run);
        let ranking_path = run_path.join(format!("ranking_{ranking_type}.json"));
        let ranking = read_ranking(&ranking_path)?;
        if ranking_key.is_empty() {
            ranking_key = ranking.metric_key.clone();
        }
        let resolution = build_run_resolution(&run_path, &ranking).map_err(|err| {
            println!(
                "/!\\ some predictions are not found in the run {}, needs investigation",
                run_path.display()
            );
            println!("{err}\n");
            err
        })?;
        resolutions.insert(run.clone(), resolution);
        let model_entries: Vec<(String, f64)> = ranking
            .scores
            .iter()
            .map(|(name, score)| (name.clone(), *score))
            .collect();
        for (idx, (model_name, score)) in model_entries.iter().enumerate() {
            rows.push((
                run.clone(),
                format!("ranked_{idx}_unrelaxed_{model_name}.pdb"),
                *score,
            ));
        }
    }

    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut global_rank: Vec<i32> = Vec::with_capacity(rows.len());
    let mut current_rank = 0;
    let mut prev_score: Option<f64> = None;
    for (idx, row) in rows.iter().enumerate() {
        if prev_score.map_or(true, |score| score != row.2) {
            current_rank = (idx + 1) as i32;
            prev_score = Some(row.2);
        }
        global_rank.push(current_rank);
    }

    let parameters: Vec<String> = rows.iter().map(|row| row.0.clone()).collect();
    let files: Vec<String> = rows.iter().map(|row| row.1.clone()).collect();
    let scores: Vec<f64> = rows.iter().map(|row| row.2).collect();

    let mut df = DataFrame::new(vec![
        Series::new("global_rank", global_rank),
        Series::new(&ranking_key, scores),
        Series::new("parameters", parameters),
        Series::new("file", files),
    ])?;

    let mut file = File::create(runs_path.join("ranking.csv"))?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut df)?;

    env::set_var("POLARS_FMT_TABLE_FORMATTING", "NOTHING");
    env::set_var("POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES", "1");
    env::set_var("POLARS_FMT_TABLE_CELL_ALIGNMENT", "RIGHT");
    //env::set_var("POLARS_FMT_STR_LEN", "10");

    println!("{df:?}");
    Ok((ranking_key, resolutions))
}

fn create_global_ranking(
    runs_path: &Path,
    runs: &[String],
    output_path: &Path,
    ranking_type: &str,
) -> Result<PredictionRunMap, Box<dyn Error>> {
    let mut map_pred_run: PredictionRunMap = IndexMap::new();
    let mut whole_ranking: IndexMap<String, f64> = IndexMap::new();
    let mut ranking_key = String::new();

    if let Err(err) = fs::create_dir_all(output_path) {
        if err.kind() != ErrorKind::AlreadyExists {
            return Err(Box::new(err));
        }
    }

    for run in runs {
        let run_path = runs_path.join(run);
        let ranking_path = run_path.join(format!("ranking_{ranking_type}.json"));
        let ranking = read_ranking(&ranking_path)?;
        if ranking_key.is_empty() {
            ranking_key = ranking.metric_key.clone();
        }
        let order = if !ranking.order.is_empty() {
            ranking.order.clone()
        } else {
            ranking.scores.keys().cloned().collect()
        };
        for pred in order.iter() {
            let prefixed = format!("{run}_{pred}");
            map_pred_run.insert(prefixed.clone(), run.clone());
            if let Some(score) = ranking.scores.get(pred) {
                whole_ranking.insert(prefixed, *score);
            }
        }
    }

    let mut sorted_predictions: Vec<(String, f64)> =
        whole_ranking.iter().map(|(k, v)| (k.clone(), *v)).collect();
    sorted_predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let order: Vec<String> = sorted_predictions.iter().map(|(k, _)| k.clone()).collect();

    let mut scores_map = serde_json::Map::new();
    for (pred, score) in sorted_predictions {
        scores_map.insert(pred, Value::from(score));
    }
    let mut global_ranking = serde_json::Map::new();
    global_ranking.insert(ranking_key.clone(), Value::Object(scores_map));
    global_ranking.insert(
        "order".to_string(),
        Value::Array(order.iter().map(|v| Value::String(v.clone())).collect()),
    );

    let ranking_path = output_path.join(format!("ranking_{ranking_type}.json"));
    let mut file = File::create(&ranking_path)?;
    file.write_all(serde_json::to_string_pretty(&Value::Object(global_ranking))?.as_bytes())?;
    Ok(map_pred_run)
}

fn move_and_rename(
    runs_path: &Path,
    pred_run_map: &PredictionRunMap,
    resolutions: &IndexMap<String, RunResolution>,
    output_path: &Path,
    include_pickles: bool,
    include_rank: bool,
) -> Result<(), Box<dyn Error>> {
    let ranking_path = output_path.join("ranking_debug.json");
    let ranking = read_ranking(&ranking_path)?;
    let global_order = ranking.order;
    let global_scores = ranking.scores;

    let mut models: Vec<i32> = Vec::new();
    let mut runs: Vec<String> = Vec::new();
    let mut predictions: Vec<String> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    let mut mapped_names: Vec<String> = Vec::new();

    for (idx, prediction) in global_order.iter().enumerate() {
        let run_name = pred_run_map.get(prediction).ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::NotFound,
                "Prediction run mapping missing",
            )) as Box<dyn Error>
        })?;
        let run_path = runs_path.join(run_name);
        if idx == 0 {
            let features_old = run_path.join("features.pkl");
            let features_new = output_path.join("features.pkl");
            if let Err(_) = fs::copy(&features_old, &features_new) {
                println!("features.pkl not found in {run_name} run");
            }
        }

        let model_start = prediction.find("_model").ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::InvalidData,
                "Prediction name missing model suffix",
            )) as Box<dyn Error>
        })?;
        let model_name = prediction[(model_start + 1)..].to_string();
        let run_resolution = resolutions.get(run_name).ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::NotFound,
                "Run resolution missing",
            )) as Box<dyn Error>
        })?;
        let actual_name = run_resolution.model_to_actual.get(&model_name).ok_or_else(|| {
            Box::new(IoError::new(
                ErrorKind::NotFound,
                "Prediction file mapping missing",
            )) as Box<dyn Error>
        })?;
        let old_pdb_path = run_path.join(actual_name);
        let mut new_pdb_path = output_path.join(format!("{prediction}.pdb"));

        if include_rank {
            new_pdb_path = output_path.join(format!("ranked_{idx}_{prediction}.pdb"));
            models.push(idx as i32);
            runs.push(prediction[..model_start].to_string());
            predictions.push(prediction.clone());
            let score = global_scores.get(prediction).cloned().unwrap_or(0.0);
            scores.push(score);
            mapped_names.push(
                new_pdb_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            );
        }

        fs::copy(&old_pdb_path, &new_pdb_path)?;

        if include_pickles {
            let mut pickles_path = run_path.clone();
            if pickles_path.join("light_pkl").is_dir() {
                pickles_path = pickles_path.join("light_pkl");
            }
            let local_pred_name = model_name.clone();
            let old_pickle_path = pickles_path.join(format!("result_{local_pred_name}.pkl"));
            let new_pickle_path = output_path.join(format!("{prediction}.pkl"));
            fs::copy(&old_pickle_path, &new_pickle_path)?;
        }
    }

    if include_rank {
        let mut df = DataFrame::new(vec![
            Series::new("model", models),
            Series::new("run", runs),
            Series::new("prediction", predictions),
            Series::new("score", scores),
            Series::new("mapped_name", mapped_names),
        ])?;
        let mut file = File::create(output_path.join("map.csv"))?;
        CsvWriter::new(&mut file)
            .include_header(true)
            .finish(&mut df)?;
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
    let sequence_name = runs_path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| String::from(""));

    let mut ignored = vec![
        String::from("all_pdbs"),
        String::from("all_runs"),
        String::from("msas"),
        String::from("msas_colabfold"),
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
    let (_ranking_key, resolutions) = rank_all(runs_path, &runs, output_path, "debug")?;

    if !only_ranking {
        let pred_run_map = create_global_ranking(runs_path, &runs, output_path, "debug")?;
        println!("Gathering {sequence_name}'s runs");
        if include_pickles {
            println!("Pickle files are also included in the gathering.");
        }
        move_and_rename(
            runs_path,
            &pred_run_map,
            &resolutions,
            output_path,
            include_pickles,
            include_rank,
        )?;
    } else if output_path.exists() {
        fs::remove_dir_all(output_path)?;
    }
    Ok(())
}
