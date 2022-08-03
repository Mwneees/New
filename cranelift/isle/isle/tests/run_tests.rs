//! Helper for autogenerated unit tests.

use cranelift_isle::error::Result;
use cranelift_isle::{compile, lexer, parser};
use std::default::Default;

fn build(filename: &str) -> Result<String> {
    let lexer = lexer::Lexer::from_files(vec![filename])?;
    let defs = parser::parse(lexer)?;
    compile::compile(&defs, &Default::default())
}

pub fn run_pass(filename: &str) {
    assert!(build(filename).is_ok());
}

pub fn run_fail(filename: &str) {
    assert!(build(filename).is_err());
}

pub fn run_link(isle_filename: &str) {
    let tempdir = tempfile::tempdir().unwrap();
    let code = build(isle_filename).unwrap();

    let isle_filename_base = std::path::Path::new(isle_filename)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let isle_generated_code = tempdir
        .path()
        .to_path_buf()
        .join(isle_filename_base.clone() + ".rs");
    std::fs::write(isle_generated_code, code).unwrap();

    let rust_filename = isle_filename.replace(".isle", "").to_string() + "_main.rs";
    let rust_filename_base = std::path::Path::new(&rust_filename).file_name().unwrap();
    let rust_driver = tempdir.path().to_path_buf().join(&rust_filename_base);
    println!("copying {} to {:?}", rust_filename, rust_driver);
    std::fs::copy(&rust_filename, &rust_driver).unwrap();

    let output = tempdir.path().to_path_buf().join("out");

    let mut rustc = std::process::Command::new("rustc")
        .arg(&rust_driver)
        .arg("-o")
        .arg(output)
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    assert!(rustc.wait().unwrap().success());
}

// Generated by build.rs.
include!(concat!(env!("OUT_DIR"), "/isle_tests.rs"));
