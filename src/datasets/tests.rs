use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    vec::Vec,
};

use flate2::{Compression, write::GzEncoder};
use tempfile::tempdir;
use zip::write::SimpleFileOptions;

use super::{
    ArchiveMode, CacheMode, DatasetFetchOptions, LotusSmiles, ZINC20_EXPECTED_RECORD_COUNT,
    Zinc20Smiles,
    coconut::{COCONUT_SMILES, CoconutSmiles},
    fetch::{default_dataset_cache_dir, gunzip_file, untar_gzip_file, unzip_file},
    massspecgym::MASS_SPEC_GYM_SMILES,
    pubchem::{PUBCHEM_SMILES, PubChemSmiles},
    reader::{DatasetSmilesIter, DatasetSmilesRecordIter},
    source::{DatasetCollectionSource, DatasetSource, SmilesDatasetRecordSource},
    types::{DatasetArtifact, DatasetCollectionArtifact, DatasetCompression, DatasetError},
    zinc20::ZINC20_SMILES,
};

fn write_zinc20_tar_gzip(path: &Path, chunk_dir: &str, contents: &[u8]) {
    let file = File::create(path).unwrap();
    let encoder = GzEncoder::new(file, Compression::default());
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_size(
        u64::try_from(contents.len())
            .unwrap_or_else(|_| unreachable!("fixture length fits into u64")),
    );
    header.set_mode(0o600);
    header.set_cksum();
    builder.append_data(&mut header, format!("{chunk_dir}/smiles_all_01.txt"), contents).unwrap();
    let encoder = builder.into_inner().unwrap();
    encoder.finish().unwrap();
}

fn write_zip(path: &Path, file_name: &str, contents: &[u8]) {
    let file = File::create(path).unwrap();
    let mut archive = zip::ZipWriter::new(file);

    archive.start_file(file_name, SimpleFileOptions::default()).unwrap();

    archive.write_all(contents).unwrap();
    archive.finish().unwrap();
}

#[test]
fn pubchem_smiles_metadata_matches_current_upstream_layout() {
    let dataset = PubChemSmiles;

    assert_eq!(dataset.id(), "pubchem-smiles");
    assert_eq!(dataset.file_name(), "CID-SMILES.gz");
    assert_eq!(dataset.extracted_file_name(), "CID-SMILES");
    assert_eq!(dataset.compression(), DatasetCompression::Gzip);
    assert!(dataset.url().contains("pubchem/Compound/Extras/CID-SMILES.gz"));
}

#[test]
fn massspecgym_smiles_metadata_matches_current_upstream_layout() {
    assert_eq!(MASS_SPEC_GYM_SMILES.id(), "massspecgym-smiles");
    assert_eq!(MASS_SPEC_GYM_SMILES.file_name(), "MassSpecGym.tsv");
    assert_eq!(MASS_SPEC_GYM_SMILES.extracted_file_name(), "MassSpecGym.tsv");
    assert_eq!(MASS_SPEC_GYM_SMILES.compression(), DatasetCompression::None);
    assert!(MASS_SPEC_GYM_SMILES.url().contains("/MassSpecGym.tsv"));
}

#[test]
fn zinc20_smiles_metadata_matches_current_upstream_layout() {
    assert_eq!(ZINC20_SMILES.id(), "zinc20-smiles");
    assert_eq!(ZINC20_SMILES.first_chunk(), 1);
    assert_eq!(ZINC20_SMILES.last_chunk(), 20);
    assert_eq!(ZINC20_EXPECTED_RECORD_COUNT, 1_006_651_037);

    let files = ZINC20_SMILES.files();
    assert_eq!(files.len(), 20);
    assert_eq!(files[0].file_name(), "ZINC20_smiles_chunk_1.tar.gz");
    assert_eq!(files[0].extracted_file_name(), "ZINC20_smiles_chunk_1");
    assert_eq!(files[0].compression(), DatasetCompression::TarGzip);
    assert!(files[0].url().contains("zinc20-ML/smiles/ZINC20_smiles_chunk_1.tar.gz"));
    assert_eq!(files[19].file_name(), "ZINC20_smiles_chunk_20.tar.gz");
}

#[test]
fn zinc20_chunk_range_validates_selection() {
    let chunk = Zinc20Smiles::chunk(7).unwrap();
    assert_eq!(chunk.first_chunk(), 7);
    assert_eq!(chunk.last_chunk(), 7);

    let range = Zinc20Smiles::chunk_range(3, 5).unwrap();
    assert_eq!(range.files().len(), 3);

    assert!(Zinc20Smiles::chunk(0).is_err());
    assert!(Zinc20Smiles::chunk_range(5, 3).is_err());
    assert!(Zinc20Smiles::chunk_range(1, 21).is_err());
}

#[test]
fn default_fetch_options_keep_compressed_cache_behavior() {
    let options = DatasetFetchOptions::default();

    assert_eq!(options.cache_mode, CacheMode::UseCache);
    assert_eq!(options.archive_mode, ArchiveMode::KeepCompressed);
    assert!(options.cache_dir.is_none());
}

#[test]
fn default_dataset_cache_dir_has_stable_suffix() {
    let cache_dir = default_dataset_cache_dir();

    assert!(cache_dir.ends_with(PathBuf::from("smiles-rs").join("datasets")));
}

#[test]
fn gunzip_file_materializes_plaintext_copy() {
    let directory = tempdir().unwrap();
    let compressed_path = directory.path().join("sample.txt.gz");
    let decompressed_path = directory.path().join("sample.txt");

    {
        let file = File::create(&compressed_path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(b"cid\tsmiles\n1\tCCO\n").unwrap();
        encoder.finish().unwrap();
    }

    gunzip_file(&compressed_path, &decompressed_path).unwrap();

    assert_eq!(fs::read(&decompressed_path).unwrap(), b"cid\tsmiles\n1\tCCO\n");
}

#[test]
fn gunzip_file_removes_the_partial_output_when_decompression_fails() {
    let directory = tempdir().unwrap();
    let compressed_path = directory.path().join("sample.txt.gz");
    let decompressed_path = directory.path().join("sample.txt");
    let partial_path = directory.path().join("sample.txt.part");

    {
        let file = File::create(&compressed_path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(&vec![b'A'; 64 * 1024]).unwrap();
        encoder.finish().unwrap();
    }

    let truncated = File::options().write(true).open(&compressed_path).unwrap();
    truncated.set_len(64).unwrap();
    drop(truncated);

    match gunzip_file(&compressed_path, &decompressed_path) {
        Err(DatasetError::Io { .. }) => {}
        Ok(_) => panic!("expected a truncated archive to fail"),
        Err(error) => panic!("unexpected error: {error}"),
    }

    assert!(!partial_path.exists());
    assert!(!decompressed_path.exists());
}

#[test]
fn pubchem_and_massspecgym_constants_are_usable_dataset_handles() {
    assert_eq!(PUBCHEM_SMILES.id(), "pubchem-smiles");
    assert_eq!(MASS_SPEC_GYM_SMILES.id(), "massspecgym-smiles");
    assert_eq!(ZINC20_SMILES.id(), "zinc20-smiles");
}

#[test]
fn dataset_smiles_iterator_is_send() {
    fn assert_send<T: Send>() {}

    assert_send::<DatasetSmilesIter>();
}

#[test]
fn pubchem_smiles_iterator_streams_smiles_from_gzip_records() {
    let directory = tempdir().unwrap();
    let compressed_path = directory.path().join("CID-SMILES.gz");

    {
        let file = File::create(&compressed_path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(b"1\tCCO\n2\tc1ccccc1\n").unwrap();
        encoder.finish().unwrap();
    }

    let artifact = DatasetArtifact {
        dataset_id: "pubchem-smiles",
        path: compressed_path.clone(),
        compressed_path: Some(compressed_path),
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    let smiles =
        DatasetSmilesIter::from_records(DatasetSmilesRecordIter::for_pubchem(&artifact).unwrap())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

    assert_eq!(smiles, ["CCO", "c1ccccc1"]);
}

#[test]
fn pubchem_record_iterator_streams_identifiers_and_smiles() {
    let directory = tempdir().unwrap();
    let compressed_path = directory.path().join("CID-SMILES.gz");

    {
        let file = File::create(&compressed_path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(b"123\tCCO\n456\tc1ccccc1\n").unwrap();
        encoder.finish().unwrap();
    }

    let artifact = DatasetArtifact {
        dataset_id: "pubchem-smiles",
        path: compressed_path.clone(),
        compressed_path: Some(compressed_path),
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    let records = DatasetSmilesRecordIter::for_pubchem(&artifact)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(records[0].id(), "123");
    assert_eq!(records[0].smiles(), "CCO");
    assert_eq!(records[1].id(), "456");
    assert_eq!(records[1].smiles(), "c1ccccc1");
}

#[test]
fn massspecgym_smiles_iterator_uses_smiles_tsv_column() {
    let directory = tempdir().unwrap();
    let dataset_path = directory.path().join("MassSpecGym.tsv");

    fs::write(&dataset_path, "spec_id\tname\tsmiles\n1\tethanol\tCCO\n2\tbenzene\tc1ccccc1\n")
        .unwrap();

    let artifact = DatasetArtifact {
        dataset_id: "massspecgym-smiles",
        path: dataset_path,
        compressed_path: None,
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    let smiles = DatasetSmilesIter::from_records(
        DatasetSmilesRecordIter::for_mass_spec_gym(&artifact).unwrap(),
    )
    .collect::<Result<Vec<_>, _>>()
    .unwrap();

    assert_eq!(smiles, ["CCO", "c1ccccc1"]);
}

#[test]
fn zinc20_record_iterator_streams_records_from_extracted_chunk() {
    let directory = tempdir().unwrap();
    let extracted_path = directory.path().join("ZINC20_smiles_chunk_1");

    fs::create_dir_all(&extracted_path).unwrap();
    fs::write(
        extracted_path.join("smiles_all_01.txt"),
        "CCO ZINC000000000001_1\nc1ccccc1 ZINC000000000002_1\n",
    )
    .unwrap();

    let artifact = DatasetCollectionArtifact {
        dataset_id: "zinc20-smiles",
        paths: vec![extracted_path],
        compressed_paths: Vec::new(),
        was_downloaded: false,
        was_extracted: false,
    };

    let records = DatasetSmilesRecordIter::for_zinc20(&artifact)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(records[0].smiles(), "CCO");
    assert_eq!(records[0].id(), "ZINC000000000001_1");
    assert_eq!(records[1].smiles(), "c1ccccc1");
    assert_eq!(records[1].id(), "ZINC000000000002_1");
}

#[test]
fn zinc20_record_iterator_rejects_malformed_rows() {
    let directory = tempdir().unwrap();
    let extracted_path = directory.path().join("ZINC20_smiles_chunk_1");

    fs::create_dir_all(&extracted_path).unwrap();
    fs::write(extracted_path.join("smiles_all_01.txt"), "CCO\n").unwrap();

    let artifact = DatasetCollectionArtifact {
        dataset_id: "zinc20-smiles",
        paths: vec![extracted_path],
        compressed_paths: Vec::new(),
        was_downloaded: false,
        was_extracted: false,
    };

    match DatasetSmilesRecordIter::for_zinc20(&artifact).unwrap().next() {
        Some(Err(DatasetError::Format { dataset_id: "zinc20-smiles", line_number: 1, .. })) => {}
        other => panic!("unexpected result: {other:?}"),
    }
}

#[test]
fn untar_gzip_file_materializes_zinc20_chunk_directory() {
    let directory = tempdir().unwrap();
    let compressed_path = directory.path().join("ZINC20_smiles_chunk_1.tar.gz");
    let extracted_path = directory.path().join("ZINC20_smiles_chunk_1");

    write_zinc20_tar_gzip(&compressed_path, "ZINC20_smiles_chunk_1", b"CCO ZINC000000000001_1\n");

    untar_gzip_file(&compressed_path, &extracted_path).unwrap();

    assert_eq!(
        fs::read_to_string(extracted_path.join("smiles_all_01.txt")).unwrap(),
        "CCO ZINC000000000001_1\n"
    );
}

#[test]
fn massspecgym_smiles_iterator_requires_smiles_header_column() {
    let directory = tempdir().unwrap();
    let dataset_path = directory.path().join("MassSpecGym.tsv");

    fs::write(&dataset_path, "spec_id\tname\n1\tethanol\n").unwrap();

    let artifact = DatasetArtifact {
        dataset_id: "massspecgym-smiles",
        path: dataset_path,
        compressed_path: None,
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    match DatasetSmilesRecordIter::for_mass_spec_gym(&artifact) {
        Ok(_) => panic!("expected a missing smiles header to fail"),
        Err(DatasetError::Format { dataset_id: "massspecgym-smiles", line_number: 1, .. }) => {}
        Err(error) => panic!("unexpected error: {error}"),
    }
}

#[test]
fn fetch_dataset_reuses_cached_uncompressed_file() {
    let directory = tempdir().unwrap();
    let dataset_directory = directory.path().join("massspecgym-smiles");

    fs::create_dir_all(&dataset_directory).unwrap();

    let dataset_path = dataset_directory.join("MassSpecGym.tsv");
    fs::write(&dataset_path, "spec_id\tsmiles\n1\tCCO\n").unwrap();

    let artifact = MASS_SPEC_GYM_SMILES
        .fetch_with_options(&DatasetFetchOptions {
            cache_dir: Some(directory.path().to_path_buf()),
            cache_mode: CacheMode::UseCache,
            archive_mode: ArchiveMode::KeepCompressed,
        })
        .unwrap();

    assert_eq!(artifact.path(), dataset_path);
    assert_eq!(artifact.compressed_path(), Some(dataset_path.as_path()));
    assert_eq!(artifact.decompressed_path(), None);
    assert!(!artifact.was_downloaded());
    assert!(!artifact.was_decompressed());
}

#[test]
fn fetch_dataset_decompresses_cached_gzip_file() {
    let directory = tempdir().unwrap();
    let dataset_directory = directory.path().join("pubchem-smiles");

    fs::create_dir_all(&dataset_directory).unwrap();

    let compressed_path = dataset_directory.join("CID-SMILES.gz");

    {
        let file = File::create(&compressed_path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(b"1\tCCO\n").unwrap();
        encoder.finish().unwrap();
    }

    let decompressed_path = dataset_directory.join("CID-SMILES");

    let artifact = PUBCHEM_SMILES
        .fetch_with_options(&DatasetFetchOptions {
            cache_dir: Some(directory.path().to_path_buf()),
            cache_mode: CacheMode::UseCache,
            archive_mode: ArchiveMode::Decompress,
        })
        .unwrap();

    assert_eq!(artifact.path(), decompressed_path);
    assert_eq!(artifact.compressed_path(), Some(compressed_path.as_path()));
    assert_eq!(artifact.decompressed_path(), Some(decompressed_path.as_path()));
    assert!(!artifact.was_downloaded());
    assert!(artifact.was_decompressed());

    assert_eq!(fs::read_to_string(decompressed_path).unwrap(), "1\tCCO\n");
}

#[test]
fn zip_dataset_keep_compressed_and_decompress_use_correct_paths() {
    struct TestZipDataset;

    impl DatasetSource for TestZipDataset {
        fn id(&self) -> &'static str {
            "test-zip"
        }

        fn url(&self) -> &'static str {
            "https://example.invalid/archive.zip"
        }

        fn file_name(&self) -> &'static str {
            "archive.zip"
        }

        fn extracted_file_name(&self) -> &'static str {
            "payload.txt"
        }

        fn compression(&self) -> DatasetCompression {
            DatasetCompression::Zip
        }
    }

    let directory = tempdir().unwrap();
    let dataset_directory = directory.path().join("test-zip");
    fs::create_dir_all(&dataset_directory).unwrap();

    let archive_path = dataset_directory.join("archive.zip");
    let extracted_path = dataset_directory.join("payload.txt");

    {
        let file = File::create(&archive_path).unwrap();
        let mut archive = zip::ZipWriter::new(file);

        archive.start_file("payload.txt", zip::write::SimpleFileOptions::default()).unwrap();
        archive.write_all(b"hello from zip\n").unwrap();
        archive.finish().unwrap();
    }

    let keep_compressed = TestZipDataset
        .fetch_with_options(&DatasetFetchOptions {
            cache_dir: Some(directory.path().to_path_buf()),
            cache_mode: CacheMode::UseCache,
            archive_mode: ArchiveMode::KeepCompressed,
        })
        .unwrap();

    assert_eq!(keep_compressed.path(), archive_path);
    assert_eq!(keep_compressed.compressed_path(), Some(archive_path.as_path()));
    assert_eq!(keep_compressed.decompressed_path(), None);

    let decompressed = TestZipDataset
        .fetch_with_options(&DatasetFetchOptions {
            cache_dir: Some(directory.path().to_path_buf()),
            cache_mode: CacheMode::UseCache,
            archive_mode: ArchiveMode::Decompress,
        })
        .unwrap();

    assert_eq!(decompressed.path(), extracted_path);
    assert_eq!(fs::read(&extracted_path).unwrap(), b"hello from zip\n");
}

#[test]
fn unzip_file_rejects_a_directory_named_like_the_payload() {
    let directory = tempdir().unwrap();
    let archive_path = directory.path().join("archive.zip");
    let extracted_path = directory.path().join("payload.txt");

    {
        let file = File::create(&archive_path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        archive.add_directory("payload.txt", zip::write::SimpleFileOptions::default()).unwrap();
        archive.finish().unwrap();
    }

    match unzip_file(&archive_path, &extracted_path) {
        Err(DatasetError::Io { path, source }) => {
            assert_eq!(path, extracted_path);
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        Ok(_) => panic!("expected a directory entry to be rejected"),
        Err(error) => panic!("unexpected error: {error}"),
    }

    assert!(!extracted_path.exists());
}
#[test]
fn unzip_file_removes_the_partial_output_when_extraction_fails() {
    let directory = tempdir().unwrap();
    let archive_path = directory.path().join("archive.zip");
    let extracted_path = directory.path().join("payload.txt");
    let partial_path = directory.path().join("payload.txt.part");

    {
        let file = File::create(&archive_path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        archive.start_file("payload.txt", zip::write::SimpleFileOptions::default()).unwrap();
        archive.write_all(&vec![b'A'; 64 * 1024]).unwrap();
        archive.finish().unwrap();
    }

    let mut bytes = fs::read(&archive_path).unwrap();
    let corrupted = bytes.len() / 2;
    bytes[corrupted] ^= 0xff;
    fs::write(&archive_path, &bytes).unwrap();

    match unzip_file(&archive_path, &extracted_path) {
        Err(DatasetError::Io { .. }) => {}
        Ok(_) => panic!("expected a corrupted entry to fail"),
        Err(error) => panic!("unexpected error: {error}"),
    }

    assert!(!partial_path.exists());
    assert!(!extracted_path.exists());
}

#[test]
fn coconut_smiles_metadata_matches_current_upstream_layout() {
    let dataset = CoconutSmiles;

    assert_eq!(dataset.id(), "coconut-smiles");
    assert_eq!(dataset.file_name(), "coconut_csv-08-2026.zip");
    assert_eq!(dataset.extracted_file_name(), "coconut_csv-08-2026.csv");
    assert_eq!(dataset.compression(), DatasetCompression::Zip);
    assert!(dataset.url().contains("/2026-08/coconut_csv-08-2026.zip"));
}

#[test]
fn unzip_file_materializes_coconut_csv() {
    let directory = tempdir().unwrap();

    let compressed_path = directory.path().join("coconut_csv-08-2026.zip");
    let extracted_path = directory.path().join("coconut_csv-08-2026.csv");

    write_zip(&compressed_path, "coconut_csv-08-2026.csv", b"identifier,smiles\nCNP000001,CCO\n");

    unzip_file(&compressed_path, &extracted_path).unwrap();

    assert_eq!(fs::read_to_string(&extracted_path).unwrap(), "identifier,smiles\nCNP000001,CCO\n");
}

#[test]
fn unzip_file_requires_expected_file_name() {
    let directory = tempdir().unwrap();

    let compressed_path = directory.path().join("coconut_csv-08-2026.zip");
    let extracted_path = directory.path().join("coconut_csv-08-2026.csv");

    write_zip(&compressed_path, "something_else.csv", b"identifier,smiles\nCNP000001,CCO\n");

    match unzip_file(&compressed_path, &extracted_path) {
        Err(DatasetError::Io { path, source }) => {
            assert_eq!(path, extracted_path);
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        Ok(_) => panic!("expected missing ZIP entry to fail"),
        Err(error) => panic!("unexpected error: {error}"),
    }
}

#[test]
fn coconut_record_iterator_uses_identifier_and_canonical_smiles_columns() {
    let directory = tempdir().unwrap();

    let dataset_path = directory.path().join("coconut_csv-08-2026.csv");

    fs::write(
        &dataset_path,
        "identifier,canonical_smiles,name\n\
         CNP000001,CCO,\"ethanol, ethyl alcohol\"\n\
         CNP000002,c1ccccc1,\"benzene\naromatic compound\"\n",
    )
    .unwrap();

    let artifact = DatasetArtifact {
        dataset_id: "coconut-smiles",
        path: dataset_path,
        compressed_path: None,
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    let records = DatasetSmilesRecordIter::for_coconut(&artifact)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(records.len(), 2);

    assert_eq!(records[0].id(), "CNP000001");
    assert_eq!(records[0].smiles(), "CCO");

    assert_eq!(records[1].id(), "CNP000002");
    assert_eq!(records[1].smiles(), "c1ccccc1");
}

#[test]
fn coconut_record_iterator_requires_the_expected_header_columns() {
    let directory = tempdir().unwrap();
    let dataset_path = directory.path().join("coconut_csv-08-2026.csv");

    fs::write(&dataset_path, "identifier,smiles,name\nCNP000001,CCO,ethanol\n").unwrap();

    let artifact = DatasetArtifact {
        dataset_id: "coconut-smiles",
        path: dataset_path,
        compressed_path: None,
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    match DatasetSmilesRecordIter::for_coconut(&artifact) {
        Err(DatasetError::Format { dataset_id: "coconut-smiles", line_number: 1, message }) => {
            assert!(message.contains("canonical_smiles"), "unexpected message: {message}");
        }
        Ok(_) => panic!("expected a header without canonical_smiles to fail"),
        Err(error) => panic!("unexpected error: {error}"),
    }
}

#[test]
fn coconut_records_reject_an_archive_that_was_kept_compressed() {
    let directory = tempdir().unwrap();
    let dataset_directory = directory.path().join("coconut-smiles");

    fs::create_dir_all(&dataset_directory).unwrap();

    let compressed_path = dataset_directory.join("coconut_csv-08-2026.zip");
    write_zip(
        &compressed_path,
        "coconut_csv-08-2026.csv",
        b"identifier,canonical_smiles\nCNP000001,CCO\n",
    );

    match COCONUT_SMILES.iter_records_with_options(&DatasetFetchOptions {
        cache_dir: Some(directory.path().to_path_buf()),
        cache_mode: CacheMode::UseCache,
        archive_mode: ArchiveMode::KeepCompressed,
    }) {
        Err(DatasetError::InvalidSelection { dataset_id: "coconut-smiles", message }) => {
            assert!(message.contains("Decompress"), "unexpected message: {message}");
        }
        Ok(_) => panic!("expected reading records from the archive to be refused"),
        Err(error) => panic!("unexpected error: {error}"),
    }
}

#[test]
fn lotus_smiles_metadata_matches_current_upstream_layout() {
    let dataset = LotusSmiles;

    assert_eq!(dataset.id(), "lotus-smiles");
    assert_eq!(dataset.file_name(), "Lotus.smi");
    assert_eq!(dataset.extracted_file_name(), "Lotus.smi");
    assert_eq!(dataset.compression(), DatasetCompression::None);
    assert!(dataset.url().contains("/download/smiles"));
}

#[test]
fn lotus_record_iterator_streams_smiles_and_identifiers() {
    let directory = tempdir().unwrap();

    let dataset_path = directory.path().join("Lotus.smi");

    fs::write(&dataset_path, "CCO LTS0000001\nc1ccccc1 LTS0000002\n").unwrap();

    let artifact = DatasetArtifact {
        dataset_id: "lotus-smiles",
        path: dataset_path,
        compressed_path: None,
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    let records = DatasetSmilesRecordIter::for_lotus(&artifact)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(records.len(), 2);

    assert_eq!(records[0].smiles(), "CCO");
    assert_eq!(records[0].id(), "LTS0000001");

    assert_eq!(records[1].smiles(), "c1ccccc1");
    assert_eq!(records[1].id(), "LTS0000002");
}

#[test]
fn lotus_record_iterator_rejects_malformed_rows() {
    let directory = tempdir().unwrap();

    let dataset_path = directory.path().join("Lotus.smi");
    fs::write(&dataset_path, "CCO\n").unwrap();

    let artifact = DatasetArtifact {
        dataset_id: "lotus-smiles",
        path: dataset_path,
        compressed_path: None,
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    match DatasetSmilesRecordIter::for_lotus(&artifact).unwrap().next() {
        Some(Err(DatasetError::Format { dataset_id: "lotus-smiles", line_number: 1, .. })) => {}
        other => panic!("unexpected result: {other:?}"),
    }
}

#[test]
fn lotus_record_iterator_reads_the_tab_separated_upstream_layout() {
    let directory = tempdir().unwrap();

    let dataset_path = directory.path().join("Lotus.smi");
    fs::write(&dataset_path, "CCO\tLTS0000001\nc1ccccc1\tLTS0000002\n").unwrap();

    let artifact = DatasetArtifact {
        dataset_id: "lotus-smiles",
        path: dataset_path,
        compressed_path: None,
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    let records = DatasetSmilesRecordIter::for_lotus(&artifact)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(records.len(), 2);

    assert_eq!(records[0].smiles(), "CCO");
    assert_eq!(records[0].id(), "LTS0000001");

    assert_eq!(records[1].smiles(), "c1ccccc1");
    assert_eq!(records[1].id(), "LTS0000002");
}

#[test]
fn lotus_record_iterator_rejects_a_row_with_a_third_field() {
    let directory = tempdir().unwrap();

    let dataset_path = directory.path().join("Lotus.smi");
    fs::write(&dataset_path, "CCO LTS0000001 junk\n").unwrap();

    let artifact = DatasetArtifact {
        dataset_id: "lotus-smiles",
        path: dataset_path,
        compressed_path: None,
        decompressed_path: None,
        was_downloaded: false,
        was_decompressed: false,
    };

    match DatasetSmilesRecordIter::for_lotus(&artifact).unwrap().next() {
        Some(Err(DatasetError::Format { dataset_id: "lotus-smiles", line_number: 1, .. })) => {}
        other => panic!("unexpected result: {other:?}"),
    }
}
