use alloc::{borrow::ToOwned, string::String};
use std::{
    io::{self, IsTerminal, Read},
    path::Path,
    time::Duration,
};

use indicatif::{ProgressBar, ProgressStyle};

const PROGRESS_TICK_INTERVAL: Duration = Duration::from_millis(100);

const SPINNER_PROGRESS_TEMPLATE: &str =
    "{spinner:.green} {msg} [{elapsed_precise}] {bytes} ({bytes_per_sec})";

const BYTE_PROGRESS_TEMPLATE: &str = "{spinner:.green} {msg} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})";

pub(crate) fn progress_label(action: &str, path: &Path) -> String {
    let file_name = path
        .file_name()
        .map_or_else(|| "dataset".into(), |name| name.to_string_lossy().into_owned());
    format!("{action} {file_name}")
}

pub(crate) fn new_byte_progress_bar(total_bytes: Option<u64>, message: &str) -> ProgressBar {
    if !io::stderr().is_terminal() {
        return ProgressBar::hidden();
    }

    if let Some(total_bytes) = total_bytes.filter(|&bytes| bytes > 0) {
        let progress_bar = ProgressBar::new(total_bytes);
        progress_bar.set_style(
            ProgressStyle::with_template(BYTE_PROGRESS_TEMPLATE)
                .unwrap_or_else(|_| unreachable!("progress template is static and valid")),
        );
        progress_bar.set_message(message.to_owned());
        progress_bar
    } else {
        let progress_bar = ProgressBar::new_spinner();
        progress_bar.set_style(
            ProgressStyle::with_template(SPINNER_PROGRESS_TEMPLATE)
                .unwrap_or_else(|_| unreachable!("progress template is static and valid")),
        );
        progress_bar.set_message(message.to_owned());
        progress_bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
        progress_bar
    }
}

pub(crate) struct ProgressReader<R> {
    inner: R,
    progress_bar: ProgressBar,
}

impl<R> ProgressReader<R> {
    pub(crate) fn new(inner: R, progress_bar: ProgressBar) -> Self {
        Self { inner, progress_bar }
    }
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
        let bytes_read = self.inner.read(buf)?;
        if bytes_read > 0 {
            self.progress_bar.inc(u64::try_from(bytes_read).unwrap_or(u64::MAX));
        }
        Ok(bytes_read)
    }
}
