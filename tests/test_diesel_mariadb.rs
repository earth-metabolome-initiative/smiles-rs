//! MariaDB integration for Diesel SMILES text columns.

use diesel::{Connection, MariadbConnection as TestConnection};

mod diesel_common;

#[test]
#[ignore = "requires a live MariaDB server and MARIADB_DATABASE_URL"]
fn mariadb_stores_and_loads_smiles_text() {
    let database_url =
        std::env::var("MARIADB_DATABASE_URL").expect("set MARIADB_DATABASE_URL for MariaDB");
    let mut connection = TestConnection::establish(&database_url).expect("MariaDB connection");
    diesel_common::check_text_storage(&mut connection);
}
