//! Validate parser coverage across large downloaded SMILES corpora.

#![cfg(feature = "datasets")]

use std::{
    env, io,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use smiles_rs::prelude::{
    ArchiveMode, CacheMode, DatasetFetchOptions, DatasetSmilesRecord, PUBCHEM_SMILES, Smiles,
    SmilesDatasetRecordSource, WildcardSmiles, ZINC20_EXPECTED_RECORD_COUNT, ZINC20_SMILES,
    Zinc20Smiles,
};

#[test]
#[ignore = "This test streams the whole PubChem CID-SMILES corpus and validates parsing in parallel."]
fn validate_pubchem_parsing_corpus() -> Result<(), Box<dyn std::error::Error>> {
    require_release_build("PubChem")?;

    let limit = env_usize("PUBCHEM_PARSE_VALIDATE_LIMIT");
    let mut options = DatasetFetchOptions {
        cache_mode: cache_mode_from_env("PUBCHEM_VALIDATE_REDOWNLOAD"),
        archive_mode: ArchiveMode::KeepCompressed,
        ..DatasetFetchOptions::default()
    };
    if let Ok(cache_dir) = env::var("PUBCHEM_VALIDATE_CACHE_DIR") {
        options.cache_dir = Some(PathBuf::from(cache_dir));
    }

    validate_records("pubchem", PUBCHEM_SMILES.iter_records_with_options(&options)?, limit, None)
}

#[test]
#[ignore = "This test streams the ZINC20-ML SMILES corpus and validates parsing in parallel."]
fn validate_zinc20_parsing_corpus() -> Result<(), Box<dyn std::error::Error>> {
    require_release_build("ZINC20")?;

    let dataset = zinc20_dataset_from_env()?;
    let limit = env_usize("ZINC20_VALIDATE_LIMIT");
    let mut options = DatasetFetchOptions {
        cache_mode: cache_mode_from_env("ZINC20_VALIDATE_REDOWNLOAD"),
        archive_mode: ArchiveMode::Decompress,
        ..DatasetFetchOptions::default()
    };
    if let Ok(cache_dir) = env::var("ZINC20_VALIDATE_CACHE_DIR") {
        options.cache_dir = Some(PathBuf::from(cache_dir));
    }

    let expected_records =
        limit.or_else(|| (dataset == ZINC20_SMILES).then_some(ZINC20_EXPECTED_RECORD_COUNT));
    validate_records(
        "zinc20",
        dataset.iter_records_with_options(&options)?,
        limit,
        expected_records,
    )
}

fn validate_records(
    label: &str,
    records: impl Iterator<Item = Result<DatasetSmilesRecord, smiles_rs::DatasetError>> + Send,
    limit: Option<usize>,
    expected_records: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let record_count = limit.unwrap_or(usize::MAX);
    let progress_bar =
        expected_records.map_or_else(ProgressBar::new_spinner, |records| {
            ProgressBar::new(u64::try_from(records).unwrap_or_else(|_| {
                unreachable!("usize always fits into u64 on supported targets")
            }))
        });
    progress_bar.set_style(
        ProgressStyle::with_template(
            "{wide_bar} {pos}/{len} [{elapsed_precise}<{eta_precise}] {per_sec} {msg}",
        )
        .unwrap_or_else(|_| unreachable!("progress template is static and valid"))
        .progress_chars("=>-"),
    );
    progress_bar.set_message(format!("{label} parsing"));

    let started = Instant::now();
    let validated = AtomicUsize::new(0);
    let wildcards = AtomicUsize::new(0);

    let validation_result =
        records.take(record_count).par_bridge().try_for_each(|record| -> Result<(), String> {
            let record = record.map_err(|error| error.to_string())?;
            parse_record(label, &record, &wildcards)?;
            validated.fetch_add(1, Ordering::Relaxed);
            progress_bar.inc(1);
            Ok(())
        });

    if let Err(error) = validation_result {
        progress_bar.abandon_with_message(format!("{label} parsing failed"));
        return Err(io::Error::other(error).into());
    }

    progress_bar.finish_with_message(format!("{label} parsing complete"));
    let validated = validated.load(Ordering::Relaxed);
    let wildcards = wildcards.load(Ordering::Relaxed);
    let elapsed = started.elapsed();
    let rate = records_per_second(validated, elapsed);
    eprintln!(
        "{label}: completed records={validated} wildcards={wildcards} elapsed={:.1}s rate={rate}/s",
        elapsed.as_secs_f64()
    );

    Ok(())
}

fn parse_record(
    label: &str,
    record: &DatasetSmilesRecord,
    wildcards: &AtomicUsize,
) -> Result<(), String> {
    let smiles_text = record.smiles();
    if smiles_text.contains('*') {
        wildcards.fetch_add(1, Ordering::Relaxed);
        smiles_text.parse::<WildcardSmiles>().map_err(|error| {
            format!(
                "{label}_id={} failed to parse wildcard SMILES:\n{}",
                record.id(),
                error.render(smiles_text)
            )
        })?;
    } else {
        smiles_text.parse::<Smiles>().map_err(|error| {
            format!(
                "{label}_id={} failed to parse SMILES:\n{}",
                record.id(),
                error.render(smiles_text)
            )
        })?;
    }
    Ok(())
}

fn zinc20_dataset_from_env() -> Result<Zinc20Smiles, Box<dyn std::error::Error>> {
    let Ok(raw) = env::var("ZINC20_VALIDATE_CHUNKS") else {
        return Ok(ZINC20_SMILES);
    };
    if let Some((first, last)) = raw.split_once('-') {
        return Ok(Zinc20Smiles::chunk_range(first.parse()?, last.parse()?)?);
    }
    Ok(Zinc20Smiles::chunk(raw.parse()?)?)
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|raw| raw.parse().ok())
}

fn cache_mode_from_env(name: &str) -> CacheMode {
    env::var(name).map_or(CacheMode::UseCache, |raw| {
        match raw.as_str() {
            "1" | "true" | "TRUE" | "yes" | "YES" => CacheMode::Redownload,
            _ => CacheMode::UseCache,
        }
    })
}

fn require_release_build(dataset_name: &str) -> Result<(), io::Error> {
    if cfg!(debug_assertions) {
        return Err(io::Error::other(format!(
            "run this ignored {dataset_name} validation with --release to avoid prohibitively slow debug runs"
        )));
    }
    Ok(())
}

fn records_per_second(records: usize, elapsed: std::time::Duration) -> usize {
    let nanos = elapsed.as_nanos();
    if nanos == 0 {
        return 0;
    }

    let records =
        u128::try_from(records).unwrap_or_else(|_| unreachable!("usize always fits into u128"));
    let per_second = (records.saturating_mul(1_000_000_000).saturating_add(nanos / 2)) / nanos;
    usize::try_from(per_second).unwrap_or(usize::MAX)
}
