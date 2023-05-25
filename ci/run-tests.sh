#!/bin/bash

cargo test \
    --features "test-programs/test_programs" \
    --features "test-programs/test_programs_http" \
    --features wasi-threads \
    --workspace \
    --exclude 'wasmtime-wasi-*' \
    --exclude wasi-crypto \
    --exclude wasi-tests \
    --exclude wasi-http-tests \
    --exclude command-tests \
    --exclude reactor-tests \
    $@
