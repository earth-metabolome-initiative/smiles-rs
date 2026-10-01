use alloc::{
    borrow::ToOwned,
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;

use super::{
    types::{DatasetArtifact, DatasetCollectionArtifact, DatasetError, DatasetSmilesRecord},
    zinc20::collect_zinc20_smiles_paths,
};

/// A streaming iterator over SMILES strings extracted from a dataset file.
pub struct DatasetSmilesIter {
    inner: DatasetSmilesRecordIter,
}

/// A streaming iterator over SMILES records extracted from a dataset file.
pub struct DatasetSmilesRecordIter {
    dataset_id: &'static str,
    source: RecordSource,
}

/// The two shapes a dataset's records arrive in.
enum RecordSource {
    Lines(LineRecords),
    Csv(CsvRecords),
}

/// Records read line by line from one or more plain or gzipped text files.
struct LineRecords {
    paths: Vec<PathBuf>,
    next_path_index: usize,
    current: Option<DatasetReader>,
    parser: LineParser,
    line_number: usize,
    line_buffer: String,
}

/// Records read from a single CSV file with a header row.
struct CsvRecords {
    path: PathBuf,
    reader: csv::Reader<File>,
    record: csv::StringRecord,
    id_column: usize,
    smiles_column: usize,
    line_number: usize,
}

struct DatasetReader {
    path: PathBuf,
    reader: Box<dyn BufRead + Send>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum LineParser {
    PubChem,
    MassSpecGym { smiles_column: usize },
    SmilesThenId,
}

impl DatasetSmilesIter {
    pub(crate) fn from_records(inner: DatasetSmilesRecordIter) -> Self {
        Self { inner }
    }
}

impl Iterator for DatasetSmilesIter {
    type Item = Result<String, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|record| record.map(DatasetSmilesRecord::into_smiles))
    }
}

impl DatasetSmilesRecordIter {
    pub(crate) fn for_pubchem(artifact: &DatasetArtifact) -> Result<Self, DatasetError> {
        Self::from_artifact(artifact, LineParser::PubChem)
    }

    pub(crate) fn for_mass_spec_gym(artifact: &DatasetArtifact) -> Result<Self, DatasetError> {
        let dataset_id = artifact.dataset_id();
        let path = artifact.path();
        let mut reader = open_text_reader(path)?;
        let mut header = String::new();
        let bytes_read = reader
            .read_line(&mut header)
            .map_err(|source| DatasetError::Io { path: path.to_path_buf(), source })?;
        if bytes_read == 0 {
            return Err(DatasetError::Format {
                dataset_id,
                line_number: 1,
                message: "expected a TSV header row with a smiles column".into(),
            });
        }

        let smiles_column = header
            .trim_end_matches(['\r', '\n'])
            .split('\t')
            .position(|field| field.eq_ignore_ascii_case("smiles"))
            .ok_or_else(|| {
                DatasetError::Format {
                    dataset_id,
                    line_number: 1,
                    message: "expected a TSV header containing a smiles column".into(),
                }
            })?;

        Ok(Self {
            dataset_id,
            source: RecordSource::Lines(LineRecords {
                paths: Vec::new(),
                next_path_index: 0,
                current: Some(DatasetReader { path: path.to_path_buf(), reader }),
                parser: LineParser::MassSpecGym { smiles_column },
                line_number: 1,
                line_buffer: String::new(),
            }),
        })
    }

    pub(crate) fn for_zinc20(artifact: &DatasetCollectionArtifact) -> Result<Self, DatasetError> {
        let mut paths = Vec::new();
        for path in artifact.paths() {
            collect_zinc20_smiles_paths(artifact.dataset_id(), path, &mut paths)?;
        }
        if paths.is_empty() {
            return Err(DatasetError::Format {
                dataset_id: artifact.dataset_id(),
                line_number: 1,
                message: "expected at least one extracted ZINC20 smiles_all_*.txt file".into(),
            });
        }

        Ok(Self {
            dataset_id: artifact.dataset_id(),
            source: RecordSource::Lines(LineRecords {
                paths,
                next_path_index: 0,
                current: None,
                parser: LineParser::SmilesThenId,
                line_number: 0,
                line_buffer: String::new(),
            }),
        })
    }

    pub(crate) fn for_coconut(artifact: &DatasetArtifact) -> Result<Self, DatasetError> {
        let dataset_id = artifact.dataset_id();
        let path = artifact.decompressed_path().unwrap_or_else(|| artifact.path());
        if path.extension().is_some_and(|extension| extension == "zip") {
            return Err(DatasetError::InvalidSelection {
                dataset_id,
                message: "reading records needs the extracted CSV: fetch with \
                          ArchiveMode::Decompress or ArchiveMode::KeepBoth"
                    .into(),
            });
        }

        let file = File::open(path)
            .map_err(|source| DatasetError::Io { path: path.to_path_buf(), source })?;
        let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(file);
        let headers = reader.headers().map_err(|error| csv_error(dataset_id, path, 1, error))?;
        let id_column = column_index(dataset_id, headers, "identifier")?;
        let smiles_column = column_index(dataset_id, headers, "canonical_smiles")?;

        Ok(Self {
            dataset_id,
            source: RecordSource::Csv(CsvRecords {
                path: path.to_path_buf(),
                reader,
                record: csv::StringRecord::new(),
                id_column,
                smiles_column,
                line_number: 1,
            }),
        })
    }

    pub(crate) fn for_lotus(artifact: &DatasetArtifact) -> Result<Self, DatasetError> {
        Self::from_artifact(artifact, LineParser::SmilesThenId)
    }

    fn from_artifact(artifact: &DatasetArtifact, parser: LineParser) -> Result<Self, DatasetError> {
        let path = artifact.path();
        Ok(Self {
            dataset_id: artifact.dataset_id(),
            source: RecordSource::Lines(LineRecords {
                paths: Vec::new(),
                next_path_index: 0,
                current: Some(DatasetReader {
                    path: path.to_path_buf(),
                    reader: open_text_reader(path)?,
                }),
                parser,
                line_number: 0,
                line_buffer: String::new(),
            }),
        })
    }
}

impl Iterator for DatasetSmilesRecordIter {
    type Item = Result<DatasetSmilesRecord, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.source {
            RecordSource::Lines(lines) => lines.next_record(self.dataset_id),
            RecordSource::Csv(csv) => csv.next_record(self.dataset_id),
        }
    }
}

impl LineRecords {
    fn next_record(
        &mut self,
        dataset_id: &'static str,
    ) -> Option<Result<DatasetSmilesRecord, DatasetError>> {
        loop {
            if self.current.is_none() {
                match self.open_next_reader()? {
                    Ok(()) => {}
                    Err(error) => return Some(Err(error)),
                }
            }

            self.line_buffer.clear();
            let read_result = {
                let current =
                    self.current.as_mut().unwrap_or_else(|| unreachable!("current reader is open"));
                current.reader.read_line(&mut self.line_buffer)
            };
            match read_result {
                Ok(0) => {
                    self.current = None;
                    continue;
                }
                Ok(_) => {
                    self.line_number += 1;
                }
                Err(source) => {
                    let path = self
                        .current
                        .as_ref()
                        .map_or_else(PathBuf::new, |current| current.path.clone());
                    return Some(Err(DatasetError::Io { path, source }));
                }
            }

            let line = self.line_buffer.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                continue;
            }

            return Some(parse_smiles_record(dataset_id, self.line_number, self.parser, line));
        }
    }

    fn open_next_reader(&mut self) -> Option<Result<(), DatasetError>> {
        let path = self.paths.get(self.next_path_index)?.clone();
        self.next_path_index += 1;
        match open_text_reader(&path) {
            Ok(reader) => {
                self.current = Some(DatasetReader { path, reader });
                self.line_number = 0;
                Some(Ok(()))
            }
            Err(error) => Some(Err(error)),
        }
    }
}

impl CsvRecords {
    fn next_record(
        &mut self,
        dataset_id: &'static str,
    ) -> Option<Result<DatasetSmilesRecord, DatasetError>> {
        let fallback_line = self.line_number.saturating_add(1);
        match self.reader.read_record(&mut self.record) {
            Ok(false) => None,
            Ok(true) => {
                self.line_number = record_line(self.record.position(), fallback_line);
                Some(self.parse_record(dataset_id, self.line_number))
            }
            Err(error) => {
                self.line_number = record_line(error.position(), fallback_line);
                Some(Err(csv_error(dataset_id, &self.path, self.line_number, error)))
            }
        }
    }

    fn parse_record(
        &self,
        dataset_id: &'static str,
        line_number: usize,
    ) -> Result<DatasetSmilesRecord, DatasetError> {
        let id = self.field(self.id_column).ok_or_else(|| {
            DatasetError::Format {
                dataset_id,
                line_number,
                message: "expected a COCONUT CSV row with an identifier".into(),
            }
        })?;
        let smiles = self.field(self.smiles_column).ok_or_else(|| {
            DatasetError::Format {
                dataset_id,
                line_number,
                message: "expected a COCONUT CSV row with a canonical_smiles value".into(),
            }
        })?;
        Ok(DatasetSmilesRecord::new(id.to_owned(), smiles.to_owned()))
    }

    fn field(&self, column: usize) -> Option<&str> {
        self.record.get(column).filter(|field| !field.is_empty())
    }
}

fn column_index(
    dataset_id: &'static str,
    headers: &csv::StringRecord,
    column: &str,
) -> Result<usize, DatasetError> {
    headers.iter().position(|field| field.eq_ignore_ascii_case(column)).ok_or_else(|| {
        DatasetError::Format {
            dataset_id,
            line_number: 1,
            message: format!("expected a CSV header containing a {column} column"),
        }
    })
}

fn record_line(position: Option<&csv::Position>, fallback_line: usize) -> usize {
    position.and_then(|position| usize::try_from(position.line()).ok()).unwrap_or(fallback_line)
}

fn parse_smiles_record(
    dataset_id: &'static str,
    line_number: usize,
    parser: LineParser,
    line: &str,
) -> Result<DatasetSmilesRecord, DatasetError> {
    match parser {
        LineParser::PubChem => {
            let (id, smiles) = line.split_once('\t').ok_or_else(|| {
                DatasetError::Format {
                    dataset_id,
                    line_number,
                    message: "expected a CID<TAB>SMILES record".into(),
                }
            })?;
            Ok(DatasetSmilesRecord::new(id.to_owned(), smiles.to_owned()))
        }
        LineParser::MassSpecGym { smiles_column } => {
            let smiles = tsv_field(line, smiles_column).ok_or_else(|| {
                DatasetError::Format {
                    dataset_id,
                    line_number,
                    message: "expected a TSV row with a smiles column value".into(),
                }
            })?;
            let id = tsv_field(line, 0).unwrap_or("");
            Ok(DatasetSmilesRecord::new(id.to_owned(), smiles.to_owned()))
        }
        LineParser::SmilesThenId => {
            let mut fields = line.split_whitespace();
            let smiles = fields.next().ok_or_else(|| {
                DatasetError::Format {
                    dataset_id,
                    line_number,
                    message: "expected a whitespace-separated SMILES and identifier record".into(),
                }
            })?;
            let id = fields.next().ok_or_else(|| {
                DatasetError::Format {
                    dataset_id,
                    line_number,
                    message: "expected a whitespace-separated SMILES and identifier record".into(),
                }
            })?;
            if fields.next().is_some() {
                return Err(DatasetError::Format {
                    dataset_id,
                    line_number,
                    message: "expected exactly two whitespace-separated fields".into(),
                });
            }
            Ok(DatasetSmilesRecord::new(id.to_owned(), smiles.to_owned()))
        }
    }
}

fn open_text_reader(path: &Path) -> Result<Box<dyn BufRead + Send>, DatasetError> {
    let file =
        File::open(path).map_err(|source| DatasetError::Io { path: path.to_path_buf(), source })?;

    if path.extension().is_some_and(|extension| extension == "gz") {
        Ok(Box::new(BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn tsv_field(line: &str, column_index: usize) -> Option<&str> {
    line.split('\t').nth(column_index)
}

fn csv_error(
    dataset_id: &'static str,
    path: &Path,
    fallback_line: usize,
    error: csv::Error,
) -> DatasetError {
    let line_number = record_line(error.position(), fallback_line);
    let message = error.to_string();
    match error.into_kind() {
        csv::ErrorKind::Io(source) => DatasetError::Io { path: path.to_path_buf(), source },
        _ => DatasetError::Format { dataset_id, line_number, message },
    }
}
