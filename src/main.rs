fn main() {
    massif::tune_allocator();
    if let Err(err) = massif::cli::run_from_env() {
        eprintln!("massif failed: {}", err);
        std::process::exit(1);
    }
}
