//! Live PostgreSQL integration for Diesel SMILES text columns.
#![cfg(feature = "diesel-postgres")]

use diesel::{Connection, PgConnection, RunQueryDsl, connection::SimpleConnection, prelude::*};
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
#[ignore = "requires a live PostgreSQL server and DATABASE_URL"]
fn postgres_stores_and_loads_smiles_text() {
    let database_url = std::env::var("DATABASE_URL").expect("set DATABASE_URL for PostgreSQL");
    let connection = &mut PgConnection::establish(&database_url).expect("PostgreSQL connection");
    connection
        .batch_execute("CREATE TEMP TABLE molecules (id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY, molecule TEXT NOT NULL)")
        .expect("create temporary molecule table");

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

    let wildcard = WildcardSmiles::from_str("*C").expect("wildcard molecule");
    diesel::insert_into(molecules::table)
        .values(molecules::molecule.eq(&wildcard))
        .execute(connection)
        .expect("insert wildcard molecule");
    let loaded_wildcard: WildcardSmiles = molecules::table
        .filter(molecules::id.eq(2))
        .select(molecules::molecule)
        .first(connection)
        .expect("read wildcard molecule");
    assert_eq!(loaded_wildcard.render(), wildcard.render());
    let strict_result = molecules::table
        .filter(molecules::id.eq(2))
        .select(molecules::molecule)
        .first::<Smiles>(connection);
    assert!(matches!(strict_result, Err(diesel::result::Error::DeserializationError(_))));
}
