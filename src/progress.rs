use indicatif::{ProgressState, ProgressStyle};
use std::fmt::Write as FmtWrite;

/// Define the style of the progress bar
pub(crate) fn default_progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
    )
    .unwrap()
    .with_key("eta", |state: &ProgressState, w: &mut dyn FmtWrite| {
        write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap();
    })
    .progress_chars("#>-")
}
