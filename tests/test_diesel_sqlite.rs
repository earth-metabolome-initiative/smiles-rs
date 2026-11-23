//! SQLite integration for Diesel SMILES text columns.
#![cfg(feature = "diesel-sqlite")]

use diesel::{
    Connection, ExpressionMethods, RunQueryDsl, SqliteConnection, connection::SimpleConnection,
    prelude::*,
};
use smiles_rs::{Smiles, WildcardSmiles};

mod schema {
    diesel::table! {
        /// Stored molecules.
        molecules (id) {
            /// Row identifier.
            id -> Integer,
            /// The SMILES text.
            molecule -> Text,
        }
    }
}
use schema::molecules;

#[test]
fn sqlite_stores_smiles_as_text_and_loads_the_graph() {
    let connection = &mut SqliteConnection::establish(":memory:").expect("SQLite connection");
    connection
        .batch_execute("CREATE TABLE molecules (id INTEGER PRIMARY KEY, molecule TEXT NOT NULL)")
        .expect("create molecule table");
    let molecule = Smiles::from_str("C/C=C\\O").expect("stereochemical molecule");

    diesel::insert_into(molecules::table)
        .values(molecules::molecule.eq(&molecule))
        .execute(connection)
        .expect("insert molecule");

    let stored: String =
        molecules::table.select(molecules::molecule).first(connection).expect("read stored text");
    assert_eq!(stored, molecule.render());
    let loaded: Smiles = molecules::table
        .select(molecules::molecule)
        .first(connection)
        .expect("read parsed molecule");
    assert_eq!(loaded, molecule);
}

#[test]
fn sqlite_rejects_invalid_text_and_supports_wildcard_molecules() {
    let connection = &mut SqliteConnection::establish(":memory:").expect("SQLite connection");
    connection
        .batch_execute("CREATE TABLE molecules (id INTEGER PRIMARY KEY, molecule TEXT NOT NULL)")
        .expect("create molecule table");
    let wildcard = WildcardSmiles::from_str("*C").expect("wildcard molecule");
    diesel::insert_into(molecules::table)
        .values(molecules::molecule.eq(&wildcard))
        .execute(connection)
        .expect("insert wildcard molecule");

    let stored: String =
        molecules::table.select(molecules::molecule).first(connection).expect("read wildcard text");
    assert_eq!(stored, wildcard.render());
    let loaded: WildcardSmiles = molecules::table
        .select(molecules::molecule)
        .first(connection)
        .expect("read wildcard molecule");
    assert_eq!(loaded.render(), stored);
    let strict_result = molecules::table.select(molecules::molecule).first::<Smiles>(connection);
    assert!(matches!(strict_result, Err(diesel::result::Error::DeserializationError(_))));

    diesel::insert_into(molecules::table)
        .values(molecules::molecule.eq("C("))
        .execute(connection)
        .expect("insert malformed text");
    let invalid_result = molecules::table
        .filter(molecules::id.eq(2))
        .select(molecules::molecule)
        .first::<WildcardSmiles>(connection);
    assert!(matches!(invalid_result, Err(diesel::result::Error::DeserializationError(_))));
}
