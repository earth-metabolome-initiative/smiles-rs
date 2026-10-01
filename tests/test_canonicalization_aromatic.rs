//! Public canonicalization behavior tests for aromatic cases.

mod canonicalization_common;

use canonicalization_common::{assert_same_canonical_group, canonical_string};

#[test]
fn canonicalize_converges_benzene_aromatic_and_kekule_forms() {
    assert_eq!(canonical_string("c1ccccc1"), canonical_string("C1=CC=CC=C1"));
}

#[test]
fn canonicalize_converges_source_backed_aromatic_equivalence_groups() {
    let groups = [
        &["FC1=CC=CC=C1O", "FC1=C(O)C=CC=C1", "Fc1ccccc1O"][..],
        &["n1ccccc1", "N1=CC=CC=C1"][..],
        &["N1C=CC=C1", "[nH]1cccc1"][..],
        &["o1cccc1", "O1C=CC=C1"][..],
        &["s1cccc1", "S1C=CC=C1"][..],
        &["[cH-]1cccc1", "[CH-]1C=CC=C1"][..],
        &["c1cc2cccccc2c1", "C1=CC2=CC=CC=CC2=C1"][..],
        &["C1=CC=CN=C1", "c1cccnc1", "n1ccccc1"][..],
        &["[H]c2c([H])c(c1c(nc(n1([H]))C(F)(F)F)c2Cl)Cl", "Clc1ccc(Cl)c2[nH]c([nH0]c21)C(F)(F)F"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}
