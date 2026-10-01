//! Exact validation checks against frozen PubChem non-aromaticity records.

use serde::Deserialize;
use smiles_rs::prelude::{AromaticityPolicy, AromaticityStatus, Smiles};

#[derive(Debug, Deserialize)]
struct PubChemNonAromaticRecord {
    cid: u64,
    smiles: String,
    atoms: usize,
    bonds: usize,
    aromatic_atom_ids: Vec<usize>,
    aromatic_bond_edges: Vec<[usize; 2]>,
    aromatic_atom_count: usize,
    aromatic_bond_count: usize,
}

fn validate_record_line_with_policy(line: &str, policy: AromaticityPolicy) -> Result<(), String> {
    let record = serde_json::from_str::<PubChemNonAromaticRecord>(line)
        .map_err(|error| format!("failed to parse corpus record JSON: {error}\nline={line}"))?;

    let smiles = record.smiles.parse::<Smiles>().map_err(|error| {
        format!("cid={} failed to parse SMILES:\n{}", record.cid, error.render(&record.smiles))
    })?;

    if smiles.nodes().len() != record.atoms {
        return Err(format!(
            "cid={} atom-count mismatch: parser={} corpus={} smiles={}",
            record.cid,
            smiles.nodes().len(),
            record.atoms,
            record.smiles
        ));
    }

    if smiles.number_of_bonds() != record.bonds {
        return Err(format!(
            "cid={} bond-count mismatch: parser={} corpus={} smiles={}",
            record.cid,
            smiles.number_of_bonds(),
            record.bonds,
            record.smiles
        ));
    }

    if !record.aromatic_atom_ids.is_empty() || record.aromatic_atom_count != 0 {
        return Err(format!(
            "cid={} corpus non-aromatic record carries aromatic atoms: ids={:?} count={} smiles={}",
            record.cid, record.aromatic_atom_ids, record.aromatic_atom_count, record.smiles
        ));
    }

    if !record.aromatic_bond_edges.is_empty() || record.aromatic_bond_count != 0 {
        return Err(format!(
            "cid={} corpus non-aromatic record carries aromatic bonds: edges={:?} count={} smiles={}",
            record.cid, record.aromatic_bond_edges, record.aromatic_bond_count, record.smiles
        ));
    }

    let aromaticity = smiles.aromaticity_assignment_for(policy);
    if aromaticity.status() == AromaticityStatus::Unsupported {
        return Err(format!(
            "cid={} aromaticity status is unsupported: parser={:?} smiles={}",
            record.cid,
            aromaticity.status(),
            record.smiles
        ));
    }

    if !aromaticity.atom_ids().is_empty() {
        return Err(format!(
            "cid={} parser produced unexpected aromatic atoms: parser={:?} smiles={}",
            record.cid,
            aromaticity.atom_ids(),
            record.smiles
        ));
    }

    if !aromaticity.bond_edges().is_empty() {
        return Err(format!(
            "cid={} parser produced unexpected aromatic bonds: parser={:?} smiles={}",
            record.cid,
            aromaticity.bond_edges(),
            record.smiles
        ));
    }

    Ok(())
}

#[test]
fn exact_pubchem_non_aromatic_record_validation_accepts_matching_empty_assignment() {
    let line = r#"{"cid":6324,"smiles":"CC","atoms":2,"bonds":1,"aromatic_atom_ids":[],"aromatic_bond_edges":[],"aromatic_atom_count":0,"aromatic_bond_count":0}"#;
    validate_record_line_with_policy(line, AromaticityPolicy::RdkitDefault)
        .expect("matching non-aromatic assignment should validate");
}

#[test]
fn exact_pubchem_non_aromatic_record_validation_rejects_unexpected_aromatic_atoms() {
    let line = r#"{"cid":241,"smiles":"c1ccccc1","atoms":6,"bonds":6,"aromatic_atom_ids":[],"aromatic_bond_edges":[],"aromatic_atom_count":0,"aromatic_bond_count":0}"#;
    let error = validate_record_line_with_policy(line, AromaticityPolicy::RdkitDefault)
        .expect_err("benzene should not validate as non-aromatic");
    assert!(error.contains("unexpected aromatic atoms"), "{error}");
}

#[test]
fn exact_pubchem_mdl_non_aromatic_record_validation_accepts_imidazole() {
    let line = r#"{"cid":795,"smiles":"c1ncc[nH]1","atoms":5,"bonds":5,"aromatic_atom_ids":[],"aromatic_bond_edges":[],"aromatic_atom_count":0,"aromatic_bond_count":0}"#;
    validate_record_line_with_policy(line, AromaticityPolicy::RdkitMdl)
        .expect("imidazole should validate as non-aromatic under MDL");
}

#[test]
fn exact_pubchem_simple_non_aromatic_record_validation_accepts_tropylium() {
    let line = r#"{"cid":7066,"smiles":"[CH+]1C=CC=CC=C1","atoms":7,"bonds":7,"aromatic_atom_ids":[],"aromatic_bond_edges":[],"aromatic_atom_count":0,"aromatic_bond_count":0}"#;
    validate_record_line_with_policy(line, AromaticityPolicy::RdkitSimple)
        .expect("tropylium should validate as non-aromatic under Simple");
}

#[test]
fn exact_pubchem_simple_non_aromatic_record_validation_accepts_cid_672() {
    let line = r#"{"cid":672,"smiles":"CC1=C(N(C2=NC(=O)NC(=O)C2=N1)CC(C(C(CO)O)O)O)C","atoms":23,"bonds":24,"aromatic_atom_ids":[],"aromatic_bond_edges":[],"aromatic_atom_count":0,"aromatic_bond_count":0}"#;
    validate_record_line_with_policy(line, AromaticityPolicy::RdkitSimple)
        .expect("CID 672 should validate as non-aromatic under Simple");
}
