FROM gcr.io/oss-fuzz-base/base-builder-rust
# the base image pins a nightly older than the crate's rust-version
ENV RUSTUP_TOOLCHAIN=nightly
RUN rustup toolchain install $RUSTUP_TOOLCHAIN --profile minimal
COPY . $SRC/smiles-rs
WORKDIR $SRC/smiles-rs
COPY ./.clusterfuzzlite/build.sh $SRC/
