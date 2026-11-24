//! Diesel `Text` storage for parsed SMILES graphs.

use diesel::{deserialize, serialize, sql_types::Text};

use crate::{Smiles, WildcardSmiles};

#[cfg(feature = "diesel-sqlite")]
mod sqlite {
    use alloc::string::ToString;

    use diesel::{backend::Backend, deserialize::FromSql, serialize::ToSql, sqlite::Sqlite};

    use super::{Smiles, Text, WildcardSmiles, deserialize, serialize};

    macro_rules! impl_sqlite_codec {
        ($molecule:ty) => {
            impl FromSql<Text, Sqlite> for $molecule {
                fn from_sql(
                    mut value: <Sqlite as Backend>::RawValue<'_>,
                ) -> deserialize::Result<Self> {
                    Self::from_str(value.read_text()).map_err(|error| error.to_string().into())
                }
            }

            impl ToSql<Text, Sqlite> for $molecule {
                fn to_sql<'bind>(
                    &'bind self,
                    output: &mut serialize::Output<'bind, '_, Sqlite>,
                ) -> serialize::Result {
                    output.set_value(self.render());
                    Ok(serialize::IsNull::No)
                }
            }
        };
    }

    impl_sqlite_codec!(Smiles);
    impl_sqlite_codec!(WildcardSmiles);
}

#[cfg(any(feature = "diesel-postgres", feature = "diesel-mariadb", feature = "diesel-mysql"))]
mod server {
    use alloc::string::ToString;

    use diesel::{backend::Backend, deserialize::FromSql, serialize::ToSql};

    use super::{Smiles, Text, WildcardSmiles, deserialize, serialize};

    macro_rules! impl_server_codec {
        ($molecule:ty, $backend:ty) => {
            impl FromSql<Text, $backend> for $molecule {
                fn from_sql(
                    value: <$backend as Backend>::RawValue<'_>,
                ) -> deserialize::Result<Self> {
                    let text = core::str::from_utf8(value.as_bytes())?;
                    Self::from_str(text).map_err(|error| error.to_string().into())
                }
            }

            impl ToSql<Text, $backend> for $molecule {
                fn to_sql<'bind>(
                    &'bind self,
                    output: &mut serialize::Output<'bind, '_, $backend>,
                ) -> serialize::Result {
                    <str as ToSql<Text, $backend>>::to_sql(&self.render(), &mut output.reborrow())
                }
            }
        };
    }

    #[cfg(feature = "diesel-postgres")]
    impl_server_codec!(Smiles, diesel::pg::Pg);
    #[cfg(feature = "diesel-postgres")]
    impl_server_codec!(WildcardSmiles, diesel::pg::Pg);
    #[cfg(feature = "diesel-mariadb")]
    impl_server_codec!(Smiles, diesel::mariadb::Mariadb);
    #[cfg(feature = "diesel-mariadb")]
    impl_server_codec!(WildcardSmiles, diesel::mariadb::Mariadb);
    #[cfg(feature = "diesel-mysql")]
    impl_server_codec!(Smiles, diesel::mysql::Mysql);
    #[cfg(feature = "diesel-mysql")]
    impl_server_codec!(WildcardSmiles, diesel::mysql::Mysql);
}
