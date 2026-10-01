//! Exact validation checks against frozen PubChem aromaticity records.

use serde::Deserialize;
use smiles_rs::prelude::{AromaticityPolicy, AromaticityStatus, Smiles};

#[derive(Debug, Deserialize)]
struct PubChemAromaticRecord {
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
    let record = serde_json::from_str::<PubChemAromaticRecord>(line)
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

    let aromaticity = smiles.aromaticity_assignment_for(policy);
    if aromaticity.status() == AromaticityStatus::Unsupported {
        return Err(format!(
            "cid={} aromaticity status is unsupported: parser={:?} smiles={}",
            record.cid,
            aromaticity.status(),
            record.smiles
        ));
    }

    if record.aromatic_atom_ids.len() != record.aromatic_atom_count {
        return Err(format!(
            "cid={} corpus aromatic atom-count mismatch: ids={} count={} smiles={}",
            record.cid,
            record.aromatic_atom_ids.len(),
            record.aromatic_atom_count,
            record.smiles
        ));
    }

    if record.aromatic_bond_edges.len() != record.aromatic_bond_count {
        return Err(format!(
            "cid={} corpus aromatic bond-count mismatch: edges={} count={} smiles={}",
            record.cid,
            record.aromatic_bond_edges.len(),
            record.aromatic_bond_count,
            record.smiles
        ));
    }

    if aromaticity.atom_ids() != record.aromatic_atom_ids.as_slice() {
        return Err(format!(
            "cid={} aromatic atom-id mismatch: parser={:?} corpus={:?} smiles={}",
            record.cid,
            aromaticity.atom_ids(),
            record.aromatic_atom_ids,
            record.smiles
        ));
    }

    if aromaticity.bond_edges() != record.aromatic_bond_edges.as_slice() {
        return Err(format!(
            "cid={} aromatic bond-edge mismatch: parser={:?} corpus={:?} smiles={}",
            record.cid,
            aromaticity.bond_edges(),
            record.aromatic_bond_edges,
            record.smiles
        ));
    }

    Ok(())
}

#[test]
fn exact_pubchem_record_validation_accepts_matching_assignment() {
    let line = r#"{"cid":241,"smiles":"c1ccccc1","atoms":6,"bonds":6,"aromatic_atom_ids":[0,1,2,3,4,5],"aromatic_bond_edges":[[0,1],[0,5],[1,2],[2,3],[3,4],[4,5]],"aromatic_atom_count":6,"aromatic_bond_count":6}"#;
    validate_record_line_with_policy(line, AromaticityPolicy::RdkitDefault)
        .expect("matching exact aromatic assignment should validate");
}

#[test]
fn exact_pubchem_record_validation_rejects_wrong_atom_identity_with_matching_count() {
    let line = r#"{"cid":241,"smiles":"c1ccccc1","atoms":6,"bonds":6,"aromatic_atom_ids":[0,1,2,3,4,4],"aromatic_bond_edges":[[0,1],[0,5],[1,2],[2,3],[3,4],[4,5]],"aromatic_atom_count":6,"aromatic_bond_count":6}"#;
    let error = validate_record_line_with_policy(line, AromaticityPolicy::RdkitDefault)
        .expect_err("wrong exact aromatic atom ids should fail");
    assert!(error.contains("aromatic atom-id mismatch"), "{error}");
}

#[test]
fn exact_pubchem_record_validation_rejects_wrong_bond_identity_with_matching_count() {
    let line = r#"{"cid":241,"smiles":"c1ccccc1","atoms":6,"bonds":6,"aromatic_atom_ids":[0,1,2,3,4,5],"aromatic_bond_edges":[[0,1],[0,5],[1,2],[2,3],[3,4],[3,5]],"aromatic_atom_count":6,"aromatic_bond_count":6}"#;
    let error = validate_record_line_with_policy(line, AromaticityPolicy::RdkitDefault)
        .expect_err("wrong exact aromatic bond edges should fail");
    assert!(error.contains("aromatic bond-edge mismatch"), "{error}");
}

#[test]
fn exact_pubchem_record_validation_matches_cid_142569_rdkit_oracle() {
    let line = r#"{"cid":142569,"smiles":"C1=CC2=C3C(=C1)C=CC4=CC=CC(=C4O3)C=C2","atoms":17,"bonds":20,"aromatic_atom_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16],"aromatic_bond_edges":[[0,1],[0,5],[1,2],[2,3],[2,16],[3,4],[4,5],[4,6],[6,7],[7,8],[8,9],[8,13],[9,10],[10,11],[11,12],[12,13],[12,15],[15,16]],"aromatic_atom_count":16,"aromatic_bond_count":18}"#;
    validate_record_line_with_policy(line, AromaticityPolicy::RdkitDefault)
        .expect("CID 142569 should match the frozen RDKit PubChem oracle");
}

#[test]
fn exact_pubchem_mdl_record_validation_accepts_matching_assignment() {
    let line = r#"{"cid":241,"smiles":"c1ccccc1","atoms":6,"bonds":6,"aromatic_atom_ids":[0,1,2,3,4,5],"aromatic_bond_edges":[[0,1],[0,5],[1,2],[2,3],[3,4],[4,5]],"aromatic_atom_count":6,"aromatic_bond_count":6}"#;
    validate_record_line_with_policy(line, AromaticityPolicy::RdkitMdl)
        .expect("matching exact MDL aromatic assignment should validate");
}

#[test]
fn exact_pubchem_mdl_record_validation_rejects_default_only_assignment() {
    let line = r#"{"cid":795,"smiles":"c1ncc[nH]1","atoms":5,"bonds":5,"aromatic_atom_ids":[0,1,2,3,4],"aromatic_bond_edges":[[0,1],[0,4],[1,2],[2,3],[3,4]],"aromatic_atom_count":5,"aromatic_bond_count":5}"#;
    let error = validate_record_line_with_policy(line, AromaticityPolicy::RdkitMdl)
        .expect_err("imidazole should not validate as aromatic under MDL");
    assert!(error.contains("aromatic atom-id mismatch"), "{error}");
}

#[test]
fn exact_pubchem_simple_record_validation_accepts_matching_assignment() {
    let line = r#"{"cid":241,"smiles":"c1ccccc1","atoms":6,"bonds":6,"aromatic_atom_ids":[0,1,2,3,4,5],"aromatic_bond_edges":[[0,1],[0,5],[1,2],[2,3],[3,4],[4,5]],"aromatic_atom_count":6,"aromatic_bond_count":6}"#;
    validate_record_line_with_policy(line, AromaticityPolicy::RdkitSimple)
        .expect("matching exact Simple aromatic assignment should validate");
}

#[test]
fn exact_pubchem_simple_record_validation_rejects_default_only_assignment() {
    let line = r#"{"cid":7066,"smiles":"[CH+]1C=CC=CC=C1","atoms":7,"bonds":7,"aromatic_atom_ids":[0,1,2,3,4,5,6],"aromatic_bond_edges":[[0,1],[0,6],[1,2],[2,3],[3,4],[4,5],[5,6]],"aromatic_atom_count":7,"aromatic_bond_count":7}"#;
    let error = validate_record_line_with_policy(line, AromaticityPolicy::RdkitSimple)
        .expect_err("tropylium should not validate as aromatic under Simple");
    assert!(error.contains("aromatic atom-id mismatch"), "{error}");
}

#[test]
fn exact_pubchem_simple_record_validation_matches_cid_8936_rdkit_oracle() {
    let line = r#"{"cid":8936,"smiles":"CC[C@@]1(CC2C[C@@](C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)[C@]78CCN9[C@H]7[C@@](C=CC9)([C@H]([C@@]([C@@H]8N6C)(C(=O)OC)O)C(=O)OC)CC)OC)C(=O)OC)O","atoms":59,"bonds":67,"aromatic_atom_ids":[7,8,14,15,16,17,18,19,20,21,22,23,24,25,26],"aromatic_bond_edges":[[7,8],[7,20],[8,14],[14,15],[14,19],[15,16],[16,17],[17,18],[18,19],[19,20],[21,22],[21,26],[22,23],[23,24],[24,25],[25,26]],"aromatic_atom_count":15,"aromatic_bond_count":16}"#;
    validate_record_line_with_policy(line, AromaticityPolicy::RdkitSimple)
        .expect("CID 8936 should match the frozen RDKit Simple oracle");
}
