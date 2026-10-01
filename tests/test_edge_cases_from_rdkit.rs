//! `RDKit` parser-reject regression cases.
//!
//! These SMILES strings are currently accepted by this crate, but `RDKit`
//! `2024.09.6` rejects them during parsing with `Chem.MolFromSmiles(...)`.
//!
//! Expected work:
//! - tighten the parser so these inputs return `Err`
//! - report the error at the offending token instead of silently repairing it
//! - fix the grammar hole, not just these exact strings

use smiles_rs::smiles::Smiles;

const RD_KIT_PARSER_REJECTS: &[&str] = &[
    "C==O", "C/#N", "C()", "C=(O)N", ".", "C.", "C..O", ".C", "..", "C...O", "C#(N)O", "C/(N)O",
    "C-(N)O", "C:(N)O", "C(=)=O",
];

#[test]
fn documented_rdkit_parser_rejects_should_fail_here_too() {
    let mut unexpected_accepts = Vec::new();

    for input in RD_KIT_PARSER_REJECTS {
        if let Ok(smiles) = Smiles::from_str(input) {
            unexpected_accepts.push(format!("{input:?} parsed as {smiles}"));
        }
    }

    assert!(
        unexpected_accepts.is_empty(),
        "documented RDKit parser rejects are still accepted here:\n{}",
        unexpected_accepts.join("\n")
    );
}
