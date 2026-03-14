## Cursor Cloud specific instructions

Massif is a single-crate Rust + Python (PyO3/maturin) bioinformatics tool for analysing protein structure ensembles. There are no databases, Docker containers, or external services required.

### Building and running

- **Rust CLI**: `cargo build --release` then `cargo run --release -- --help`. See `README.md` for full subcommand usage.
- **Python package**: `pip install maturin && pip install .` — this compiles the Rust code with the `python` feature via PyO3 and installs the `massif` Python module + CLI script.
- **Linting**: `cargo clippy -- -W clippy::all` (14 pre-existing warnings, 0 errors as of v0.3.1).
- **Formatting**: `cargo fmt --check` (the existing code has formatting diffs; `cargo fmt` to auto-fix).
- **Tests**: `cargo test` — the codebase currently has **no unit tests** (0 passed, 0 failed).
- **Benchmarks**: `cargo bench` — requires sample PDB files in `benches/test/structures/`; the bench file references an old crate name (`align_rs`) and will not compile without adjustment.

### Important caveats

- The Rust stable toolchain must be >= 1.85 because `clap` (v4.6) and its transitive dep `clap_lex` use the `edition2024` Cargo feature. The VM's pre-installed toolchain (1.83) is too old; the update script runs `rustup default stable` to ensure the latest stable is active.
- After `pip install .`, the `massif` script is placed in `~/.local/bin` which may not be on `PATH`. Add `export PATH="$HOME/.local/bin:$PATH"` if needed.
- All subcommands require PDB or mmCIF structure files as input. There are no built-in test fixtures in the repository.
- The `python` feature (PyO3 bindings) is only compiled when building via maturin (`pip install .`), not by default `cargo build`.
