//! Validate canonicalization across the whole PubChem `CID-SMILES` corpus.

#![cfg(feature = "datasets")]

use std::{
    env,
    fs::File,
    io::{self, BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use molecular_formulas::prelude::ChemicalFormula;
use rayon::prelude::*;
use smiles_rs::prelude::{
    ArchiveMode, CacheMode, DatasetFetchOptions, DatasetSource, PUBCHEM_SMILES, Smiles,
    WildcardSmiles,
};

#[test]
#[ignore = "This test streams the whole PubChem CID-SMILES corpus and validates canonicalization in parallel."]
fn validate_pubchem_canonicalization_corpus() -> Result<(), Box<dyn std::error::Error>> {
    require_release_build()?;

    let corpus_path = pubchem_corpus_path()?;
    let limit = env_usize("PUBCHEM_CANONICALIZATION_VALIDATE_LIMIT");
    let total_records = pubchem_record_count(&corpus_path)?;
    let record_count = limit.map_or(total_records, |limit| limit.min(total_records));

    let progress_bar = ProgressBar::new(
        u64::try_from(record_count)
            .unwrap_or_else(|_| unreachable!("usize always fits into u64 on supported targets")),
    );
    progress_bar.set_style(
        ProgressStyle::with_template(
            "{wide_bar} {pos}/{len} [{elapsed_precise}<{eta_precise}] {per_sec} {msg}",
        )
        .unwrap_or_else(|_| unreachable!("progress template is static and valid"))
        .progress_chars("=>-"),
    );
    progress_bar.set_message("canonicalization");

    let started = Instant::now();
    let validated = AtomicUsize::new(0);

    let validation_result =
        open_pubchem_corpus(&corpus_path)?.lines().take(record_count).par_bridge().try_for_each(
            |line| -> Result<(), String> {
                let line = line.map_err(|error| error.to_string())?;
                validate_record_line(&line)?;
                validated.fetch_add(1, Ordering::Relaxed);
                progress_bar.inc(1);
                Ok(())
            },
        );

    if let Err(error) = validation_result {
        progress_bar.abandon_with_message("canonicalization failed");
        return Err(io::Error::other(error).into());
    }

    progress_bar.finish_with_message("canonicalization complete");
    let validated = validated.load(Ordering::Relaxed);
    let elapsed = started.elapsed();
    let rate = records_per_second(validated, elapsed);
    eprintln!(
        "completed validated={validated} elapsed={:.1}s rate={rate}/s",
        elapsed.as_secs_f64()
    );

    Ok(())
}

#[test]
#[ignore = "This test streams the whole PubChem CID-SMILES corpus and validates strict SMILES to formula conversion in parallel."]
fn validate_pubchem_molecular_formula_corpus() -> Result<(), Box<dyn std::error::Error>> {
    require_release_build()?;

    let corpus_path = pubchem_formula_corpus_path()?;
    let limit = env_usize("PUBCHEM_FORMULA_VALIDATE_LIMIT");
    let total_records = pubchem_record_count(&corpus_path)?;
    let record_count = limit.map_or(total_records, |limit| limit.min(total_records));

    let progress_bar = ProgressBar::new(
        u64::try_from(record_count)
            .unwrap_or_else(|_| unreachable!("usize always fits into u64 on supported targets")),
    );
    progress_bar.set_style(
        ProgressStyle::with_template(
            "{wide_bar} {pos}/{len} [{elapsed_precise}<{eta_precise}] {per_sec} {msg}",
        )
        .unwrap_or_else(|_| unreachable!("progress template is static and valid"))
        .progress_chars("=>-"),
    );
    progress_bar.set_message("molecular formula");

    let started = Instant::now();
    let formulas = AtomicUsize::new(0);
    let wildcards = AtomicUsize::new(0);

    let validation_result =
        open_pubchem_corpus(&corpus_path)?.lines().take(record_count).par_bridge().try_for_each(
            |line| -> Result<(), String> {
                let line = line.map_err(|error| error.to_string())?;
                match validate_formula_record_line(&line)? {
                    FormulaRecordKind::Formula => {
                        formulas.fetch_add(1, Ordering::Relaxed);
                    }
                    FormulaRecordKind::Wildcard => {
                        wildcards.fetch_add(1, Ordering::Relaxed);
                    }
                }
                progress_bar.inc(1);
                Ok(())
            },
        );

    if let Err(error) = validation_result {
        progress_bar.abandon_with_message("molecular formula validation failed");
        return Err(io::Error::other(error).into());
    }

    progress_bar.finish_with_message("molecular formula validation complete");
    let formulas = formulas.load(Ordering::Relaxed);
    let wildcards = wildcards.load(Ordering::Relaxed);
    let elapsed = started.elapsed();
    let rate = records_per_second(record_count, elapsed);
    eprintln!(
        "completed records={record_count} formulas={formulas} skipped_wildcards={wildcards} elapsed={:.1}s rate={rate}/s",
        elapsed.as_secs_f64()
    );

    Ok(())
}

fn pubchem_record_count(path: &Path) -> Result<usize, io::Error> {
    Ok(open_pubchem_corpus(path)?.lines().count())
}

fn pubchem_corpus_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(path) = env::var("PUBCHEM_CANONICALIZATION_CORPUS") {
        return Ok(PathBuf::from(path));
    }

    let artifact = PUBCHEM_SMILES.fetch_with_options(&DatasetFetchOptions {
        cache_mode: cache_mode_from_env("PUBCHEM_VALIDATE_REDOWNLOAD"),
        archive_mode: ArchiveMode::Decompress,
        ..DatasetFetchOptions::default()
    })?;
    Ok(artifact.path().to_path_buf())
}

fn pubchem_formula_corpus_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(path) = env::var("PUBCHEM_FORMULA_CORPUS") {
        return Ok(PathBuf::from(path));
    }

    pubchem_corpus_path()
}

fn validate_record_line(line: &str) -> Result<(), String> {
    let (cid, smiles_text) = line
        .split_once('\t')
        .ok_or_else(|| format!("expected CID<TAB>SMILES record, got: {line}"))?;

    if smiles_text.contains('*') {
        let smiles = smiles_text.parse::<WildcardSmiles>().map_err(|error| {
            format!("cid={cid} failed to parse wildcard SMILES:\n{}", error.render(smiles_text))
        })?;
        validate_canonicalization(cid, smiles_text, &smiles.canonicalize())?;
    } else {
        let smiles = smiles_text.parse::<Smiles>().map_err(|error| {
            format!("cid={cid} failed to parse SMILES:\n{}", error.render(smiles_text))
        })?;
        validate_canonicalization(cid, smiles_text, &smiles.canonicalize())?;
    }

    Ok(())
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FormulaRecordKind {
    Formula,
    Wildcard,
}

fn validate_formula_record_line(line: &str) -> Result<FormulaRecordKind, String> {
    let (cid, smiles_text) = line
        .split_once('\t')
        .ok_or_else(|| format!("expected CID<TAB>SMILES record, got: {line}"))?;

    if smiles_text.contains('*') {
        smiles_text.parse::<WildcardSmiles>().map_err(|error| {
            format!("cid={cid} failed to parse wildcard SMILES:\n{}", error.render(smiles_text))
        })?;
        return Ok(FormulaRecordKind::Wildcard);
    }

    let smiles = smiles_text.parse::<Smiles>().map_err(|error| {
        format!("cid={cid} failed to parse SMILES:\n{}", error.render(smiles_text))
    })?;
    let _formula: ChemicalFormula<u32, i32> = ChemicalFormula::from(&smiles);
    Ok(FormulaRecordKind::Formula)
}

fn validate_canonicalization(
    cid: &str,
    smiles_text: &str,
    canonicalized: &impl CanonicalizedSmiles,
) -> Result<(), String> {
    if !canonicalized.is_canonical() {
        return Err(format!(
            "cid={cid} canonicalize() did not reach a canonical fixed point: {smiles_text}"
        ));
    }

    let recanonicalized = canonicalized.canonicalize();
    if canonicalized != &recanonicalized {
        return Err(format!(
            "cid={cid} canonicalize() is not idempotent:\ninput={smiles_text}\nfirst={canonicalized}\nsecond={recanonicalized}"
        ));
    }

    Ok(())
}

trait CanonicalizedSmiles: Sized + PartialEq + core::fmt::Display {
    fn is_canonical(&self) -> bool;

    fn canonicalize(&self) -> Self;
}

impl CanonicalizedSmiles for Smiles {
    fn is_canonical(&self) -> bool {
        Smiles::is_canonical(self)
    }

    fn canonicalize(&self) -> Self {
        Smiles::canonicalize(self)
    }
}

impl CanonicalizedSmiles for WildcardSmiles {
    fn is_canonical(&self) -> bool {
        WildcardSmiles::is_canonical(self)
    }

    fn canonicalize(&self) -> Self {
        WildcardSmiles::canonicalize(self)
    }
}

enum PubchemCorpusReader {
    Plain(BufReader<File>),
    Gzip(BufReader<GzDecoder<File>>),
}

impl Read for PubchemCorpusReader {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
        match self {
            Self::Plain(reader) => reader.read(buf),
            Self::Gzip(reader) => reader.read(buf),
        }
    }
}

impl BufRead for PubchemCorpusReader {
    fn fill_buf(&mut self) -> Result<&[u8], io::Error> {
        match self {
            Self::Plain(reader) => reader.fill_buf(),
            Self::Gzip(reader) => reader.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match self {
            Self::Plain(reader) => reader.consume(amt),
            Self::Gzip(reader) => reader.consume(amt),
        }
    }
}

fn open_pubchem_corpus(path: &Path) -> Result<PubchemCorpusReader, io::Error> {
    let file = File::open(path)?;
    if path.extension().is_some_and(|ext| ext == "gz") {
        Ok(PubchemCorpusReader::Gzip(BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(PubchemCorpusReader::Plain(BufReader::new(file)))
    }
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

fn require_release_build() -> Result<(), io::Error> {
    if cfg!(debug_assertions) {
        return Err(io::Error::other(
            "run this ignored PubChem validation with --release to avoid prohibitively slow debug runs",
        ));
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
