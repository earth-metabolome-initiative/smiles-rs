//! Perf regression tests for `Smiles::perceive_aromaticity_for(RdkitDefault)`.
//! Corpus from finge-rs fuzzing (2026-05-22) where pre-fix runtimes ran
//! 0.84-5.6 s on graphs of 58-89 atoms.

use std::time::{Duration, Instant};

use smiles_rs::smiles::{AromaticityPolicy, Smiles};

const PERCEPTION_BUDGET: Duration =
    if cfg!(debug_assertions) { Duration::from_secs(10) } else { Duration::from_millis(250) };

const PATHOLOGICAL_CORPUS: &[(&str, &str)] = &[
    (
        "topological_torsion oom 2785e9 (58 atoms, ~5.6 s pre-fix)",
        "bFFFFFFFFn3SS752sBS5S73B1S3=5S7B2S5S73B2(cS3n5S7B2S5S732BS3n5S7B2S5S73B27S53SnB1S5S7B33Sn2)5S7B2C5S73S53BB27n2S5S7F3-2",
    ),
    (
        "topological_torsion oom d89e1c (85 atoms, ~3.0 s pre-fix)",
        "C2CFCBBC1FB2B2nBBBFCF22BBBnF$CF11BBBI-FFB2B2nCB2B2nBBB22BBFB2B211BBC1F2B2nCBF2B2FF2B2nF2B2nCBF2B2FF2B2nFB2B2nB2B2FB2B2nB2B2FBC2",
    ),
    (
        "topological_torsion oom f6c504 (61 atoms, ~1.8 s pre-fix)",
        "F3oFF12s3FP3PS2SFF11P2F1s2SP2F1s2S1F/sSFF12s3FP3PS2SFF11P2F1s2SP2F1s2S1F/s1FS2FS1FP2Fs1FS2FS1F1SbS3#SS21",
    ),
    (
        "ecfp timeout 50af30 (73 atoms, ~1.3 s pre-fix)",
        "F789C/C-C3C7$Cn3CC7CC3nCC7nC3CF7CC3nCC7nC30C7CC3nCC7nC3CF7CC3nCCNNF8CNBCS7nC3C7CC3nCC7nC3CF7CC3nC0C7CC39CCnNCN",
    ),
    (
        "mutator_h_count timeout 32a342 (69 atoms, ~1.3 s pre-fix)",
        "C7pC02CB1CFB2N211Bs.CF2N211CF22BF/FB12N12BBsC11BBCF22BF/C0BB7c2B1B7PcF#pCBBC210CF2N11FC2BB2N211BBCFCFB2N211/C0BB7c2B1B2B[Os]=2",
    ),
    (
        "mutator_formal_charge timeout 4536b7 (72 atoms, ~0.85 s pre-fix)",
        "FF9p7FFcFFFnF8#F7pF88FF.sF#F7pF8FsBB8#F7pF8Fs8FF#F7pFBB81[Ac]F.F8#F7pF88FF.sF#F7pF8FsBB8#N7pF8Fs8FF#F7pB=81F77Cp8FF9FcF87",
    ),
    (
        "mutator_isotope timeout 90d953 (89 atoms, ~0.84 s pre-fix)",
        "S8IoNNIII4II5oI88I4IIIINNI5II5NI88IIIIIIII5NNI5NI88IINNII84NNN5=I5I8IoNI8No8N4I4IIINN5I5IINNN88IoI4II5oI88O4IINNI5IINNI48N5I8NI8",
    ),
];

fn assert_perception_within_budget(label: &str, source: &str) {
    let smiles: Smiles = source
        .parse()
        .unwrap_or_else(|err| panic!("{label}: source must parse, got {err:?}: {source}"));
    let base = smiles.kekulize_standalone().unwrap_or_else(|_| smiles.clone());

    let started = Instant::now();
    let _ = base.perceive_aromaticity_for(AromaticityPolicy::RdkitDefault);
    let elapsed = started.elapsed();

    assert!(
        elapsed <= PERCEPTION_BUDGET,
        "{label}: perceive_aromaticity_for(RdkitDefault) took {elapsed:?}, exceeds budget {PERCEPTION_BUDGET:?}\n  smiles: {source}"
    );
}

#[test]
fn perception_budget_topological_torsion_2785e9() {
    let (label, source) = PATHOLOGICAL_CORPUS[0];
    assert_perception_within_budget(label, source);
}

#[test]
fn perception_budget_topological_torsion_d89e1c() {
    let (label, source) = PATHOLOGICAL_CORPUS[1];
    assert_perception_within_budget(label, source);
}

#[test]
fn perception_budget_topological_torsion_f6c504() {
    let (label, source) = PATHOLOGICAL_CORPUS[2];
    assert_perception_within_budget(label, source);
}

#[test]
fn perception_budget_ecfp_50af30() {
    let (label, source) = PATHOLOGICAL_CORPUS[3];
    assert_perception_within_budget(label, source);
}

#[test]
fn perception_budget_mutator_h_count_32a342() {
    let (label, source) = PATHOLOGICAL_CORPUS[4];
    assert_perception_within_budget(label, source);
}

#[test]
fn perception_budget_mutator_formal_charge_4536b7() {
    let (label, source) = PATHOLOGICAL_CORPUS[5];
    assert_perception_within_budget(label, source);
}

#[test]
fn perception_budget_mutator_isotope_90d953() {
    let (label, source) = PATHOLOGICAL_CORPUS[6];
    assert_perception_within_budget(label, source);
}
