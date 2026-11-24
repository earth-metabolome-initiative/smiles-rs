//! SQLite integration for Diesel SMILES text columns.

use diesel::{Connection, SqliteConnection as TestConnection};

mod diesel_common;

#[test]
fn sqlite_stores_and_loads_smiles_text() {
    let mut connection = TestConnection::establish(":memory:").expect("SQLite connection");
    diesel_common::check_text_storage(&mut connection);
}
