//! MySQL integration for Diesel SMILES text columns.

use diesel::{Connection, MysqlConnection as TestConnection};

mod diesel_common;

#[test]
#[ignore = "requires a live MySQL server and MYSQL_DATABASE_URL"]
fn mysql_stores_and_loads_smiles_text() {
    let database_url =
        std::env::var("MYSQL_DATABASE_URL").expect("set MYSQL_DATABASE_URL for MySQL");
    let mut connection = TestConnection::establish(&database_url).expect("MySQL connection");
    diesel_common::check_text_storage(&mut connection);
}
