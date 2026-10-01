#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedImplicitHydrogenAtom {
    pub atom: &'static str,
    pub implicit_hydrogens: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct ImplicitHydrogenCase {
    pub name: &'static str,
    pub smiles: &'static str,
    pub note: &'static str,
    pub atoms: &'static [ExpectedImplicitHydrogenAtom],
}

const fn atom(atom: &'static str, implicit_hydrogens: u8) -> ExpectedImplicitHydrogenAtom {
    ExpectedImplicitHydrogenAtom { atom, implicit_hydrogens }
}

/// Cases that should be solvable from local bond-order sums plus SMILES
/// bracket defaults.
///
/// These are directly motivated by the `OpenSMILES` and Daylight rules for:
/// - unbracketed organic-subset atoms
/// - bracket atoms defaulting to zero implicit hydrogens
/// - wildcard atoms outside brackets carrying zero implicit hydrogens
/// - explicit hydrogens never being double-counted as implicit hydrogens
/// - bracket `H0` and `H1` spellings remaining explicit-only
///
/// This list is intentionally representative rather than combinatorially
/// exhaustive. Syntax dimensions that are orthogonal to implicit-hydrogen
/// counting, such as chirality, atom classes, or most isotope spellings, are
/// covered elsewhere in parser tests instead of being duplicated here.
pub const ORGANIC_SUBSET_CASES: &[ImplicitHydrogenCase] = &[
    ImplicitHydrogenCase {
        name: "methane",
        smiles: "C",
        note: "Bare aliphatic carbon fills to valence four.",
        atoms: &[atom("C", 4)],
    },
    ImplicitHydrogenCase {
        name: "ethane",
        smiles: "CC",
        note: "Each terminal carbon contributes one explicit single bond and carries three implicit hydrogens.",
        atoms: &[atom("C", 3), atom("C", 3)],
    },
    ImplicitHydrogenCase {
        name: "carbonyl",
        smiles: "C=O",
        note: "Carbonyl carbon fills to four, oxygen fills to two.",
        atoms: &[atom("C", 2), atom("O", 0)],
    },
    ImplicitHydrogenCase {
        name: "nitrile",
        smiles: "C#N",
        note: "Nitrile carbon carries one implicit hydrogen, nitrogen carries none.",
        atoms: &[atom("C", 1), atom("N", 0)],
    },
    ImplicitHydrogenCase {
        name: "methanol",
        smiles: "CO",
        note: "Alcohol oxygen contributes one implicit hydrogen when unbracketed.",
        atoms: &[atom("C", 3), atom("O", 1)],
    },
    ImplicitHydrogenCase {
        name: "formic-acid-skeleton",
        smiles: "C(=O)O",
        note: "The carbonyl carbon has one remaining valence; the hydroxyl oxygen has one.",
        atoms: &[atom("C", 1), atom("O", 0), atom("O", 1)],
    },
    ImplicitHydrogenCase {
        name: "isobutane",
        smiles: "CC(C)C",
        note: "The central carbon is tertiary and carries one implicit hydrogen.",
        atoms: &[atom("C", 3), atom("C", 1), atom("C", 3), atom("C", 3)],
    },
    ImplicitHydrogenCase {
        name: "dichloromethane-skeleton",
        smiles: "ClCCl",
        note: "Terminal halogens fill to valence one, leaving the carbon with two implicit hydrogens.",
        atoms: &[atom("Cl", 0), atom("C", 2), atom("Cl", 0)],
    },
    ImplicitHydrogenCase {
        name: "single-atom-nitrogen",
        smiles: "N",
        note: "Bare nitrogen fills to valence three.",
        atoms: &[atom("N", 3)],
    },
    ImplicitHydrogenCase {
        name: "single-atom-oxygen",
        smiles: "O",
        note: "Bare oxygen fills to valence two.",
        atoms: &[atom("O", 2)],
    },
    ImplicitHydrogenCase {
        name: "single-atom-chlorine",
        smiles: "Cl",
        note: "A bare halogen fills to valence one.",
        atoms: &[atom("Cl", 1)],
    },
    ImplicitHydrogenCase {
        name: "single-atom-iodine",
        smiles: "I",
        note: "A bare iodine starts from target valence one in the raw SMILES model.",
        atoms: &[atom("I", 1)],
    },
    ImplicitHydrogenCase {
        name: "wildcard",
        smiles: "*",
        note: "OpenSMILES assigns zero implicit hydrogens to an unbracketed wildcard atom.",
        atoms: &[atom("*", 0)],
    },
    ImplicitHydrogenCase {
        name: "wildcard-attached-to-carbon",
        smiles: "*C",
        note: "The wildcard remains at zero while the neighboring carbon still fills to valence four.",
        atoms: &[atom("*", 0), atom("C", 3)],
    },
    ImplicitHydrogenCase {
        name: "bracket-carbon-defaults-to-h-zero",
        smiles: "[C]",
        note: "OpenSMILES defines a missing bracket H-count as H0.",
        atoms: &[atom("[C]", 0)],
    },
    ImplicitHydrogenCase {
        name: "bracket-carbon-explicit-h-one",
        smiles: "[CH]",
        note: "`[CH]` is equivalent to `[CH1]`, so the bracket atom still gets zero implicit hydrogens.",
        atoms: &[atom("[CH]", 0)],
    },
    ImplicitHydrogenCase {
        name: "molecular-hydrogen",
        smiles: "[H][H]",
        note: "Hydrogen bound to hydrogen must be written explicitly, so neither atom gets implicit hydrogens.",
        atoms: &[atom("[H]", 0), atom("[H]", 0)],
    },
    ImplicitHydrogenCase {
        name: "hydroxide",
        smiles: "[OH-]",
        note: "Bracket hydrogen on hydroxide is explicit, not implicit.",
        atoms: &[atom("[OH-]", 0)],
    },
    ImplicitHydrogenCase {
        name: "ammonium",
        smiles: "[NH4+]",
        note: "All hydrogens are explicit inside the bracket atom.",
        atoms: &[atom("[NH4+]", 0)],
    },
];

/// Edge cases that are intentionally aligned to `RDKit`'s raw property-cache
/// behavior on SMILES-as-written input.
///
/// These are still local rules, but they go beyond the minimal `OpenSMILES`
/// organic-subset defaults. The iodine and sulfur cases are included because
/// they showed up in the `PubChem` reference sweep and are easy to regress.
pub const RAW_RDKIT_COMPAT_CASES: &[ImplicitHydrogenCase] = &[
    ImplicitHydrogenCase {
        name: "difluorochlorane-like",
        smiles: "FClF",
        note: "Raw RDKit keeps neutral F, Cl, and Br capped at valence one for implicit-hydrogen purposes.",
        atoms: &[atom("F", 0), atom("Cl", 0), atom("F", 0)],
    },
    ImplicitHydrogenCase {
        name: "chlorous-acid-skeleton",
        smiles: "Cl(=O)O",
        note: "The doubly bonded oxygen gets zero, while the terminal oxygen still carries one implicit hydrogen.",
        atoms: &[atom("Cl", 0), atom("O", 0), atom("O", 1)],
    },
    ImplicitHydrogenCase {
        name: "sulfur-trifluoride-like",
        smiles: "FS(F)F",
        note: "Raw RDKit advances sulfur to the next compatible target valence, leaving one implicit hydrogen on the central sulfur.",
        atoms: &[atom("F", 0), atom("S", 1), atom("F", 0), atom("F", 0)],
    },
    ImplicitHydrogenCase {
        name: "hypervalent-iodine-target-five",
        smiles: "CI(C)(C)C",
        note: "Raw RDKit uses the neutral iodine target sequence 1, 3, 5, so tetravalent iodine gets one implicit hydrogen.",
        atoms: &[atom("C", 3), atom("I", 1), atom("C", 3), atom("C", 3), atom("C", 3)],
    },
    ImplicitHydrogenCase {
        name: "overvalent-iodine-stops-at-zero",
        smiles: "CI(C)(C)(C)(C)C",
        note: "Once neutral iodine exceeds its highest raw target valence, the implicit hydrogen count falls back to zero.",
        atoms: &[
            atom("C", 3),
            atom("I", 0),
            atom("C", 3),
            atom("C", 3),
            atom("C", 3),
            atom("C", 3),
            atom("C", 3),
        ],
    },
    ImplicitHydrogenCase {
        name: "triiodine-chain-middle-gets-one",
        smiles: "III",
        note: "The middle iodine has bond-order sum two, so the raw reference model advances it to target valence three.",
        atoms: &[atom("I", 0), atom("I", 1), atom("I", 0)],
    },
];

/// Cases that need an aromatic policy layer on top of the raw valence table.
///
/// These are still local in the SMILES sense, but they are not just
/// `elements-rs` valence ranges:
/// - lower-case aromatic atoms
/// - aromatic heteroatom conventions
/// - explicit aromatic bond syntax using `:`
/// - bracketed aromatic atoms such as `[nH]` and `[se]`
///
/// We intentionally do not treat this list as exhaustive for aromaticity
/// validation. Impossible lowercase-aromatic inputs and sanitized chemistry
/// reinterpretation belong to a separate normalization or validation layer.
pub const AROMATIC_CASES: &[ImplicitHydrogenCase] = &[
    ImplicitHydrogenCase {
        name: "benzene",
        smiles: "c1ccccc1",
        note: "Each aromatic carbon contributes one implicit hydrogen.",
        atoms: &[
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
        ],
    },
    ImplicitHydrogenCase {
        name: "benzene-with-explicit-aromatic-bonds",
        smiles: "c1:c:c:c:c:c:1",
        note: "Explicit `:` aromatic bonds do not change the local implicit-hydrogen counts.",
        atoms: &[
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
        ],
    },
    ImplicitHydrogenCase {
        name: "pyridine",
        smiles: "n1ccccc1",
        note: "Aromatic pyridine nitrogen carries zero implicit hydrogens.",
        atoms: &[
            atom("n", 0),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
        ],
    },
    ImplicitHydrogenCase {
        name: "pyrrole",
        smiles: "[nH]1cccc1",
        note: "The aromatic nitrogen hydrogen is explicit in the bracket atom.",
        atoms: &[atom("[nH]", 0), atom("c", 1), atom("c", 1), atom("c", 1), atom("c", 1)],
    },
    ImplicitHydrogenCase {
        name: "furan",
        smiles: "o1cccc1",
        note: "Aromatic oxygen carries zero implicit hydrogens while each aromatic carbon contributes one.",
        atoms: &[atom("o", 0), atom("c", 1), atom("c", 1), atom("c", 1), atom("c", 1)],
    },
    ImplicitHydrogenCase {
        name: "selenophene-bracketed",
        smiles: "[se]1cccc1",
        note: "Bracket aromatic selenium is explicit H0 unless a hydrogen count is written in the bracket.",
        atoms: &[atom("[se]", 0), atom("c", 1), atom("c", 1), atom("c", 1), atom("c", 1)],
    },
    ImplicitHydrogenCase {
        name: "tellurophene-bracketed",
        smiles: "[te]1cccc1",
        note: "Bracket aromatic tellurium is explicit H0 unless a hydrogen count is written in the bracket.",
        atoms: &[atom("[te]", 0), atom("c", 1), atom("c", 1), atom("c", 1), atom("c", 1)],
    },
    ImplicitHydrogenCase {
        name: "thiophene",
        smiles: "s1cccc1",
        note: "Aromatic sulfur carries zero implicit hydrogens while each aromatic carbon contributes one.",
        atoms: &[atom("s", 0), atom("c", 1), atom("c", 1), atom("c", 1), atom("c", 1)],
    },
    ImplicitHydrogenCase {
        name: "biphenyl",
        smiles: "c1ccccc1-c2ccccc2",
        note: "A single bond between two aromatic systems must be written explicitly, and the two ipso carbons lose their implicit hydrogens.",
        atoms: &[
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 0),
            atom("c", 0),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
        ],
    },
    ImplicitHydrogenCase {
        name: "naphthalene",
        smiles: "c1cccc2ccccc12",
        note: "The fused aromatic carbons already have valence three and therefore carry zero implicit hydrogens.",
        atoms: &[
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 0),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 0),
        ],
    },
    ImplicitHydrogenCase {
        name: "pyridinium-like",
        smiles: "[n+]1ccccc1",
        note: "A charged aromatic bracket atom still contributes zero implicit hydrogens unless an H-count is written explicitly.",
        atoms: &[
            atom("[n+]", 0),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
        ],
    },
    ImplicitHydrogenCase {
        name: "wildcard-branch-on-aromatic-ring",
        smiles: "Oc1c(*)cccc1",
        note: "OpenSMILES explicitly allows `*` in aromatic systems; the wildcard stays at H0 while substituted aromatic carbons lose their hydrogens locally.",
        atoms: &[
            atom("O", 1),
            atom("c", 0),
            atom("c", 0),
            atom("*", 0),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
            atom("c", 1),
        ],
    },
];

pub fn all_cases() -> impl Iterator<Item = &'static ImplicitHydrogenCase> {
    ORGANIC_SUBSET_CASES.iter().chain(RAW_RDKIT_COMPAT_CASES.iter()).chain(AROMATIC_CASES.iter())
}
