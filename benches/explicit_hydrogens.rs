//! Criterion benchmarks for `Smiles::with_explicit_hydrogens()`.

use std::{
    fs::File,
    hint::black_box,
    io::{BufRead, BufReader},
    path::Path,
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use smiles_rs::prelude::Smiles;

fn fixed_cases() -> [(&'static str, Smiles); 6] {
    [
        ("methanol", "CO".parse().unwrap()),
        ("ammonium", "[NH4+]".parse().unwrap()),
        ("benzene", "c1ccccc1".parse().unwrap()),
        ("cyclohexane", "C1CCCCC1".parse().unwrap()),
        ("chiral_center", "F[C@H](Cl)Br".parse().unwrap()),
        ("acetylcarnitine", "CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C".parse().unwrap()),
    ]
}

fn load_pubchem_sample(limit: usize) -> Option<Vec<Smiles>> {
    let path = Path::new("tests/CID-SMILES");
    if !path.is_file() {
        return None;
    }

    let reader = BufReader::new(File::open(path).ok()?);
    let mut smiles = Vec::with_capacity(limit);
    for line in reader.lines().take(limit) {
        let line = line.ok()?;
        let (_, smiles_text) = line.split_once('\t')?;
        smiles.push(smiles_text.parse().ok()?);
    }

    Some(smiles)
}

fn bench_fixed_cases(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("with_explicit_hydrogens_fixed");

    for (name, smiles) in fixed_cases() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(name), &smiles, |bench, smiles| {
            bench.iter(|| black_box(smiles).with_explicit_hydrogens());
        });
    }

    group.finish();
}

fn bench_pubchem_sample(criterion: &mut Criterion) {
    let Some(sample) = load_pubchem_sample(1024) else {
        return;
    };

    let mut group = criterion.benchmark_group("with_explicit_hydrogens_corpus");
    group.throughput(Throughput::Elements(sample.len() as u64));
    group.bench_function("pubchem_1024", |bench| {
        bench.iter(|| {
            let total_nodes = sample
                .iter()
                .map(|smiles| black_box(smiles).with_explicit_hydrogens().nodes().len())
                .sum::<usize>();
            black_box(total_nodes);
        });
    });
    group.finish();
}

fn explicit_hydrogens_benches(criterion: &mut Criterion) {
    bench_fixed_cases(criterion);
    bench_pubchem_sample(criterion);
}

criterion_group!(benches, explicit_hydrogens_benches);
criterion_main!(benches);
