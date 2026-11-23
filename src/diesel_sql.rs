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

#[cfg(feature = "diesel-postgres")]
mod postgres {
    use alloc::string::ToString;

    use diesel::{backend::Backend, deserialize::FromSql, pg::Pg, serialize::ToSql};

    use super::{Smiles, Text, WildcardSmiles, deserialize, serialize};

    macro_rules! impl_postgres_codec {
        ($molecule:ty) => {
            impl FromSql<Text, Pg> for $molecule {
                fn from_sql(value: <Pg as Backend>::RawValue<'_>) -> deserialize::Result<Self> {
                    let text = core::str::from_utf8(value.as_bytes())?;
                    Self::from_str(text).map_err(|error| error.to_string().into())
                }
            }

            impl ToSql<Text, Pg> for $molecule {
                fn to_sql<'bind>(
                    &'bind self,
                    output: &mut serialize::Output<'bind, '_, Pg>,
                ) -> serialize::Result {
                    <str as ToSql<Text, Pg>>::to_sql(&self.render(), &mut output.reborrow())
                }
            }
        };
    }

    impl_postgres_codec!(Smiles);
    impl_postgres_codec!(WildcardSmiles);
}
