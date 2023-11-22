#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use regex::Regex;
    use std::fs::read_to_string;
    use std::io::Error;

    use cranelift_codegen::entity::EntityRef;
    use cranelift_codegen::ir::function::FunctionParameters;
    use cranelift_codegen::ir::ExternalName;
    use cranelift_codegen::isa::zkasm;
    use cranelift_codegen::{settings, FinalizedMachReloc, FinalizedRelocTarget};
    use cranelift_wasm::{translate_module, DummyEnvironment};

    fn setup() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn generate_preamble(start_func_index: usize) -> Vec<String> {
        let mut program: Vec<String> = Vec::new();
        program.push("start:".to_string());
        program.push("  zkPC + 2 => RR".to_string());
        program.push(format!("  :JMP(function_{})", start_func_index));
        program.push("  :JMP(finalizeExecution)".to_string());
        program
    }

    fn generate_postamble() -> Vec<String> {
        let mut program: Vec<String> = Vec::new();
        // In the prover, the program always runs for a fixed number of steps (e.g. 2^23), so we
        // need an infinite loop at the end of the program to fill the execution trace to the
        // expected number of steps.
        // In the future we might need to put zero in all registers here.
        program.push("finalizeExecution:".to_string());
        program.push("  ${beforeLast()}  :JMPN(finalizeExecution)".to_string());
        program.push("                   :JMP(start)".to_string());
        program
    }

    // TODO: Relocations should be generated by a linker and/or clift itself.
    fn fix_relocs(
        code_buffer: &mut Vec<u8>,
        params: &FunctionParameters,
        relocs: &[FinalizedMachReloc],
    ) {
        let mut delta = 0i32;
        for reloc in relocs {
            let start = (reloc.offset as i32 + delta) as usize;
            let mut pos = start;
            while code_buffer[pos] != b'\n' {
                pos += 1;
                delta -= 1;
            }

            let code = if let FinalizedRelocTarget::ExternalName(ExternalName::User(name)) =
                reloc.target
            {
                let name = &params.user_named_funcs()[name];
                if name.index == 0 {
                    b"  B :ASSERT".to_vec()
                } else {
                    format!("  zkPC + 2 => RR\n  :JMP(function_{})", name.index)
                        .as_bytes()
                        .to_vec()
                }
            } else {
                b"  UNKNOWN".to_vec()
            };
            delta += code.len() as i32;

            code_buffer.splice(start..pos, code);
        }
    }

    // TODO: Labels optimization already happens in `MachBuffer`, we need to find a way to leverage
    // it.
    fn optimize_labels(code: &[&str], _func_index: usize) -> Vec<String> {
        let mut label_definition: HashMap<usize, usize> = HashMap::new();
        let mut label_uses: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut lines = Vec::new();
        for (index, line) in code.iter().enumerate() {
            let line = line.to_string();
            if line.starts_with(&"label_") {
                // Handles lines with a label marker, e.g.:
                //   label_XXX:
                let index_begin = line.rfind("_").expect("Failed to parse label index") + 1;
                let label_index: usize = line[index_begin..line.len() - 1]
                    .parse()
                    .expect("Failed to parse label index");
                label_definition.insert(label_index, index);
            } else if line.contains(&"label_") {
                // Handles lines with a jump to label, e.g.:
                // A : JMPNZ(label_XXX)
                let pos = line.find(&"label_").unwrap();
                let pos_end = pos + line[pos..].find(&")").unwrap();
                let label_index: usize = line[pos + 6..pos_end]
                    .parse()
                    .expect("Failed to parse label index");
                label_uses.entry(label_index).or_default().push(index);
            }
            lines.push(line);
        }

        let mut lines_to_delete = Vec::new();
        for (label, label_line) in label_definition {
            match label_uses.entry(label) {
                std::collections::hash_map::Entry::Occupied(uses) => {
                    if uses.get().len() == 1 {
                        let use_line = uses.get()[0];
                        if use_line + 1 == label_line {
                            lines_to_delete.push(use_line);
                            lines_to_delete.push(label_line);
                        }
                    }
                }
                std::collections::hash_map::Entry::Vacant(_) => {
                    lines_to_delete.push(label_line);
                }
            }
        }
        lines_to_delete.sort();
        lines_to_delete.reverse();
        for index in lines_to_delete {
            lines.remove(index);
        }
        lines
    }

    fn generate_zkasm(wasm_module: &[u8]) -> String {
        let flag_builder = settings::builder();
        let isa_builder = zkasm::isa_builder("zkasm-unknown-unknown".parse().unwrap());
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();
        let mut dummy_environ = DummyEnvironment::new(isa.frontend_config());
        translate_module(wasm_module, &mut dummy_environ).unwrap();

        let mut program: Vec<String> = Vec::new();
        let start_func = dummy_environ
            .info
            .start_func
            .expect("Must have a start function");
        // TODO: Preamble should be generated by a linker and/or clift itself.
        program.append(&mut generate_preamble(start_func.index()));

        let num_func_imports = dummy_environ.get_num_func_imports();
        let mut context = cranelift_codegen::Context::new();
        for (def_index, func) in dummy_environ.info.function_bodies.iter() {
            let func_index = num_func_imports + def_index.index();
            program.push(format!("function_{}:", func_index));

            let mut mem = vec![];
            context.func = func.clone();
            let compiled_code = context
                .compile_and_emit(&*isa, &mut mem, &mut Default::default())
                .unwrap();
            let mut code_buffer = compiled_code.code_buffer().to_vec();
            fix_relocs(
                &mut code_buffer,
                &func.params,
                compiled_code.buffer.relocs(),
            );

            let code = std::str::from_utf8(&code_buffer).unwrap();
            let lines: Vec<&str> = code.lines().collect();
            let mut lines = optimize_labels(&lines, func_index);
            program.append(&mut lines);

            context.clear();
        }

        program.append(&mut generate_postamble());
        program.join("\n")
    }

    fn test_module(name: &str) {
        let module_binary = wat::parse_file(format!("../zkasm_data/{name}.wat")).unwrap();
        let program = generate_zkasm(&module_binary);
        let expected =
            expect_test::expect_file![format!("../../zkasm_data/generated/{name}.zkasm")];
        expected.assert_eq(&program);
    }

    // This function asserts that none of tests generated from
    // spectest has been changed.
    fn check_spectests() -> Result<(), Error> {
        let spectests_path = "../../tests/spec_testsuite/i64.wast";
        let file_content = read_to_string(spectests_path)?;
        let re = Regex::new(
            r#"\(assert_return \(invoke \"(\w+)\"\s*((?:\([^\)]+\)\s*)+)\)\s*\(([^\)]+)\)\)"#,
        )
        .unwrap();
        let mut test_counters = HashMap::new();
        for cap in re.captures_iter(&file_content) {
            let function_name = &cap[1];
            let arguments = &cap[2];
            let expected_result = &cap[3];
            let assert_type = &expected_result[0..3];
            let count = test_counters.entry(function_name.to_string()).or_insert(0);
            *count += 1;
            let mut testcase = String::new();
            testcase.push_str(&format!("(module\n (import \"env\" \"assert_eq\" (func $assert_eq (param {}) (param {})))\n (func $main\n", assert_type, assert_type));
            testcase.push_str(&format!(
                "\t{}\n",
                arguments
                    .replace(") (", "\n\t")
                    .replace("(", "")
                    .replace(")", "")
            ));
            testcase.push_str(&format!("\ti64.{}\n", function_name));
            testcase.push_str(&format!(
                "\t{}\n\tcall $assert_eq)\n (start $main))\n",
                expected_result.trim()
            ));
            let file_name = format!(
                "../../zkasm_data/spectest/i64/{}_{}.wat",
                function_name, count
            );
            let expected_test = expect_test::expect_file![file_name];
            expected_test.assert_eq(&testcase);
        }
        Ok(())
    }

    #[test]
    fn run_spectests() {
        check_spectests().unwrap();
        let path = "../zkasm_data/spectest/i64/";
        let mut failures = 0;
        let mut count = 0;
        for entry in std::fs::read_dir(path).expect("Directory not found") {
            let entry = entry.expect("Failed to read entry");
            let file_name = entry.file_name();
            if entry.path().extension().and_then(|s| s.to_str()) != Some("wat") {
                continue;
            }
            if let Some(name) = std::path::Path::new(&file_name)
                .file_stem()
                .and_then(|s| s.to_str())
            {
                let module_binary =
                    wat::parse_file(format!("../zkasm_data/spectest/i64/{name}.wat")).unwrap();
                let expected = expect_test::expect_file![format!(
                    "../../zkasm_data/spectest/i64/generated/{name}.zkasm"
                )];
                let result = std::panic::catch_unwind(|| {
                    let program = generate_zkasm(&module_binary);
                    expected.assert_eq(&program);
                });
                count += 1;
                if let Err(err) = result {
                    failures += 1;
                    println!("{} fails with {:#?}", name, err);
                }
            }
        }
        println!("Failed {} spectests out of {}", failures, count);
    }

    macro_rules! testcases {
        { $($name:ident,)* } => {
          $(
            #[test]
            fn $name() {
                setup();
                test_module(stringify!($name));
            }
           )*
        };
    }

    testcases! {
        add,
        locals,
        locals_simple,
        counter,
        fibonacci,
        add_func,
        mul,
        i64_mul,
        lt_s,
        lt_u,
        xor,
        and,
        or,
        div,
        i64_div,
        eqz,
        ne,
        nop,
        _should_fail_unreachable,
        i32_const,
        i64_const,
        rem,
    }
}
