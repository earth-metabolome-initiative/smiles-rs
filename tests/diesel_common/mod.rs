//! Shared checks for whole molecules in Diesel text columns.

use diesel::{connection::SimpleConnection, prelude::*, result::Error::DeserializationError};
use smiles_rs::{Smiles, WildcardSmiles};

use super::TestConnection;

diesel::table! {
    /// Stored molecules.
    molecules (id) {
        /// Row identifier.
        id -> Integer,
        /// The SMILES text.
        molecule -> Text,
    }
}

pub(super) fn check_text_storage(connection: &mut TestConnection) {
    connection
        .batch_execute(
            "CREATE TEMPORARY TABLE molecules (id INTEGER PRIMARY KEY, molecule TEXT NOT NULL)",
        )
        .expect("create temporary molecule table");
    let molecule = Smiles::from_str("C/C=C\\O").expect("stereochemical molecule");
    diesel::insert_into(molecules::table)
        .values((molecules::id.eq(1), molecules::molecule.eq(&molecule)))
        .execute(connection)
        .expect("insert molecule");
    let strict_row = molecules::table.find(1).select(molecules::molecule);
    assert_eq!(
        strict_row.first::<String>(connection).expect("read stored text"),
        molecule.render()
    );
    assert_eq!(strict_row.first::<Smiles>(connection).expect("read parsed molecule"), molecule);

    let wildcard = WildcardSmiles::from_str("*C").expect("wildcard molecule");
    diesel::insert_into(molecules::table)
        .values((molecules::id.eq(2), molecules::molecule.eq(&wildcard)))
        .execute(connection)
        .expect("insert wildcard molecule");
    let wildcard_row = molecules::table.find(2).select(molecules::molecule);
    let stored = wildcard_row.first::<String>(connection).expect("read wildcard text");
    assert_eq!(stored, wildcard.render());
    assert_eq!(
        wildcard_row.first::<WildcardSmiles>(connection).expect("read wildcard molecule"),
        WildcardSmiles::from_str(&stored).expect("parse rendered wildcard molecule")
    );
    diesel::update(molecules::table.find(2))
        .set(molecules::molecule.eq("*C"))
        .execute(connection)
        .expect("store literal wildcard text");
    assert_eq!(
        wildcard_row.first::<WildcardSmiles>(connection).expect("read literal wildcard"),
        wildcard
    );
    assert!(matches!(wildcard_row.first::<Smiles>(connection), Err(DeserializationError(_))));

    diesel::insert_into(molecules::table)
        .values((molecules::id.eq(3), molecules::molecule.eq("C(")))
        .execute(connection)
        .expect("insert malformed text");
    let malformed_row = molecules::table.find(3).select(molecules::molecule);
    assert!(matches!(malformed_row.first::<Smiles>(connection), Err(DeserializationError(_))));
    assert!(matches!(
        malformed_row.first::<WildcardSmiles>(connection),
        Err(DeserializationError(_))
    ));
}
