/// glibc grows each thread's heap in small steps, and with many threads those calls queue on the kernel's memory-map lock
/// MALLOC_TOP_PAD_ in the environment still takes precedence.
/// Applied by the massif binary at startup and by the Python module when it is imported.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub fn tune_allocator() {
    if std::env::var_os("MALLOC_TOP_PAD_").is_none() {
        // SAFETY: mallopt only sets an allocator parameter; it is called before any thread starts.
        unsafe {
            libc::mallopt(libc::M_TOP_PAD, 64 << 20);
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub fn tune_allocator() {}
