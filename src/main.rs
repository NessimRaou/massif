fn main() {
    tune_allocator();
    if let Err(err) = massif::cli::run_from_env() {
        eprintln!("massif failed: {}", err);
        std::process::exit(1);
    }
}

/// glibc grows each thread's heap in small steps, and with many threads those calls queue on the kernel's memory-map lock
/// MALLOC_TOP_PAD_ in the environment still takes precedence.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn tune_allocator() {
    if std::env::var_os("MALLOC_TOP_PAD_").is_none() {
        // SAFETY: mallopt only sets an allocator parameter; it is called before any thread starts.
        unsafe {
            libc::mallopt(libc::M_TOP_PAD, 64 << 20);
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn tune_allocator() {}
