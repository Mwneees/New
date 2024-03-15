//! Test runner for `.wat` files to exercise CLIF-to-Wasm translations.

use anyhow::{bail, Context, Result};
use cranelift_codegen::ir::Function;
use cranelift_codegen::isa::TargetIsa;
use cranelift_control::ControlPlane;
use serde::de::DeserializeOwned;
use serde_derive::Deserialize;
use similar::TextDiff;
use std::{fmt::Write, path::Path};

/// Parse test configuration from the specified test, comments starting with
/// `;;!`.
pub fn parse_test_config<T>(wat: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    // The test config source is the leading lines of the WAT file that are
    // prefixed with `;;!`.
    let config_lines: Vec<_> = wat
        .lines()
        .take_while(|l| l.starts_with(";;!"))
        .map(|l| &l[3..])
        .collect();
    let config_text = config_lines.join("\n");

    toml::from_str(&config_text).context("failed to parse the test configuration")
}

/// Which kind of test is being performed.
#[derive(Default, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TestKind {
    /// Test the CLIF output, raw from translation.
    #[default]
    Clif,
    /// Compile output to machine code.
    Compile,
    /// Test the CLIF output, optimized.
    Optimize,
}

/// Assert that `wat` contains the test expectations necessary for `funcs`.
pub fn run_functions(
    path: &Path,
    wat: &str,
    isa: &dyn TargetIsa,
    kind: TestKind,
    funcs: &[Function],
) -> Result<()> {
    let mut actual = String::new();
    for func in funcs {
        match kind {
            TestKind::Compile => {
                let mut ctx = cranelift_codegen::Context::for_function(func.clone());
                ctx.set_disasm(true);
                let code = ctx
                    .compile(isa, &mut Default::default())
                    .map_err(|e| crate::pretty_anyhow_error(&e.func, e.inner))?;
                writeln!(&mut actual, "function {}:", func.name).unwrap();
                writeln!(&mut actual, "{}", code.vcode.as_ref().unwrap()).unwrap();
            }
            TestKind::Optimize => {
                let mut ctx = cranelift_codegen::Context::for_function(func.clone());
                ctx.optimize(isa, &mut ControlPlane::default())
                    .map_err(|e| crate::pretty_anyhow_error(&ctx.func, e))?;
                writeln!(&mut actual, "{}", ctx.func.display()).unwrap();
            }
            TestKind::Clif => {
                writeln!(&mut actual, "{}", func.display()).unwrap();
            }
        }
    }
    let actual = actual.trim();
    log::debug!("=== actual ===\n{actual}");

    // The test's expectation is the final comment.
    let mut expected_lines: Vec<_> = wat
        .lines()
        .rev()
        .take_while(|l| l.starts_with(";;"))
        .map(|l| {
            if l.starts_with(";; ") {
                &l[3..]
            } else {
                &l[2..]
            }
        })
        .collect();
    expected_lines.reverse();
    let expected = expected_lines.join("\n");
    let expected = expected.trim();
    log::debug!("=== expected ===\n{expected}");

    if actual == expected {
        return Ok(());
    }

    if std::env::var("CRANELIFT_TEST_BLESS").unwrap_or_default() == "1" {
        let old_expectation_line_count = wat
            .lines()
            .rev()
            .take_while(|l| l.starts_with(";;"))
            .count();
        let old_wat_line_count = wat.lines().count();
        let new_wat_lines: Vec<_> = wat
            .lines()
            .take(old_wat_line_count - old_expectation_line_count)
            .map(|l| l.to_string())
            .chain(actual.lines().map(|l| {
                if l.is_empty() {
                    ";;".to_string()
                } else {
                    format!(";; {l}")
                }
            }))
            .collect();
        let mut new_wat = new_wat_lines.join("\n");
        new_wat.push('\n');
        std::fs::write(path, new_wat)
            .with_context(|| format!("failed to write file: {}", path.display()))?;
        return Ok(());
    }

    bail!(
        "Did not get the expected CLIF translation:\n\n\
         {}\n\n\
         Note: You can re-run with the `CRANELIFT_TEST_BLESS=1` environment\n\
         variable set to update test expectations.",
        TextDiff::from_lines(expected, actual)
            .unified_diff()
            .header("expected", "actual")
    )
}
