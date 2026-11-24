//! PostgreSQL integration for Diesel SMILES text columns.

use diesel::{Connection, PgConnection as TestConnection};

mod diesel_common;

#[test]
#[ignore = "requires a live PostgreSQL server and DATABASE_URL"]
fn postgres_stores_and_loads_smiles_text() {
    let database_url = std::env::var("DATABASE_URL").expect("set DATABASE_URL for PostgreSQL");
    let mut connection = TestConnection::establish(&database_url).expect("PostgreSQL connection");
    diesel_common::check_text_storage(&mut connection);
}
