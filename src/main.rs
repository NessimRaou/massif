fn main() {
    if let Err(err) = massif::cli::run_from_env() {
        eprintln!("massif failed: {}", err);
        std::process::exit(1);
    }
}
