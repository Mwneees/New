#![cfg(not(miri))]

use super::{make_echo_component, make_echo_component_with_params, Param, Type};
use anyhow::Result;
use component_test_util::FuncExt;
use wasmtime::component::types::{self, Case, ComponentItem, Field};
use wasmtime::component::{self, Component, Linker, ResourceType, Val};
use wasmtime::Store;
use wasmtime_component_util::REALLOC_AND_FREE;

#[test]
fn primitives() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());
    let mut output = [Val::Bool(false)];

    for (input, ty, param) in [
        (Val::Bool(true), "bool", Param(Type::U8, Some(0))),
        (Val::S8(-42), "s8", Param(Type::S8, Some(0))),
        (Val::U8(42), "u8", Param(Type::U8, Some(0))),
        (Val::S16(-4242), "s16", Param(Type::S16, Some(0))),
        (Val::U16(4242), "u16", Param(Type::U16, Some(0))),
        (Val::S32(-314159265), "s32", Param(Type::I32, Some(0))),
        (Val::U32(314159265), "u32", Param(Type::I32, Some(0))),
        (Val::S64(-31415926535897), "s64", Param(Type::I64, Some(0))),
        (Val::U64(31415926535897), "u64", Param(Type::I64, Some(0))),
        (
            Val::Float32(3.14159265),
            "float32",
            Param(Type::F32, Some(0)),
        ),
        (
            Val::Float64(3.14159265),
            "float64",
            Param(Type::F64, Some(0)),
        ),
        (Val::Char('🦀'), "char", Param(Type::I32, Some(0))),
    ] {
        let component = Component::new(&engine, make_echo_component_with_params(ty, &[param]))?;
        let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
        let func = instance.get_func(&mut store, "echo").unwrap();
        func.call_and_post_return(&mut store, &[input.clone()], &mut output)?;

        assert_eq!(input, output[0]);
    }

    // Sad path: type mismatch

    let component = Component::new(
        &engine,
        make_echo_component_with_params("float64", &[Param(Type::F64, Some(0))]),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let err = func
        .call_and_post_return(&mut store, &[Val::U64(42)], &mut output)
        .unwrap_err();

    assert!(err.to_string().contains("type mismatch"), "{err}");

    // Sad path: arity mismatch (too many)

    let err = func
        .call_and_post_return(
            &mut store,
            &[Val::Float64(3.14159265), Val::Float64(3.14159265)],
            &mut output,
        )
        .unwrap_err();

    assert!(
        err.to_string().contains("expected 1 argument(s), got 2"),
        "{err}"
    );

    // Sad path: arity mismatch (too few)

    let err = func
        .call_and_post_return(&mut store, &[], &mut output)
        .unwrap_err();
    assert!(
        err.to_string().contains("expected 1 argument(s), got 0"),
        "{err}"
    );

    let err = func
        .call_and_post_return(&mut store, &output, &mut [])
        .unwrap_err();
    assert!(
        err.to_string().contains("expected 1 results(s), got 0"),
        "{err}"
    );

    Ok(())
}

#[test]
fn strings() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(&engine, make_echo_component("string", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let input = Val::String(Box::from("hello, component!"));
    let mut output = [Val::Bool(false)];
    func.call_and_post_return(&mut store, &[input.clone()], &mut output)?;
    assert_eq!(input, output[0]);

    Ok(())
}

#[test]
fn lists() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let input = ty.unwrap_list().new_val(Box::new([
        Val::U32(32343),
        Val::U32(79023439),
        Val::U32(2084037802),
    ]))?;
    let mut output = [Val::Bool(false)];
    func.call_and_post_return(&mut store, &[input.clone()], &mut output)?;

    assert_eq!(input, output[0]);

    // Sad path: type mismatch

    let err = ty
        .unwrap_list()
        .new_val(Box::new([
            Val::U32(32343),
            Val::U32(79023439),
            Val::Float32(3.14159265),
        ]))
        .unwrap_err();

    assert!(err.to_string().contains("type mismatch"), "{err}");

    Ok(())
}

#[test]
fn records() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"
                (type $c' (record
                    (field "D" bool)
                    (field "E" u32)
                ))
                (export $c "c" (type $c'))
                (type $Foo' (record
                    (field "A" u32)
                    (field "B" float64)
                    (field "C" $c)
                ))
            "#,
            &[
                Param(Type::I32, Some(0)),
                Param(Type::F64, Some(8)),
                Param(Type::U8, Some(16)),
                Param(Type::I32, Some(20)),
            ],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let inner_type = &ty.unwrap_record().fields().nth(2).unwrap().ty;
    let input = ty.unwrap_record().new_val([
        ("A", Val::U32(32343)),
        ("B", Val::Float64(3.14159265)),
        (
            "C",
            inner_type
                .unwrap_record()
                .new_val([("D", Val::Bool(false)), ("E", Val::U32(2084037802))])?,
        ),
    ])?;
    let mut output = [Val::Bool(false)];
    func.call_and_post_return(&mut store, &[input.clone()], &mut output)?;

    assert_eq!(input, output[0]);

    // Sad path: type mismatch

    let err = ty
        .unwrap_record()
        .new_val([
            ("A", Val::S32(32343)),
            ("B", Val::Float64(3.14159265)),
            (
                "C",
                inner_type
                    .unwrap_record()
                    .new_val([("D", Val::Bool(false)), ("E", Val::U32(2084037802))])?,
            ),
        ])
        .unwrap_err();

    assert!(err.to_string().contains("type mismatch"), "{err}");

    // Sad path: too many fields

    let err = ty
        .unwrap_record()
        .new_val([
            ("A", Val::U32(32343)),
            ("B", Val::Float64(3.14159265)),
            (
                "C",
                inner_type
                    .unwrap_record()
                    .new_val([("D", Val::Bool(false)), ("E", Val::U32(2084037802))])?,
            ),
            ("F", Val::Bool(true)),
        ])
        .unwrap_err();

    assert!(
        err.to_string().contains("expected 3 value(s); got 4"),
        "{err}"
    );

    // Sad path: too few fields

    let err = ty
        .unwrap_record()
        .new_val([("A", Val::U32(32343)), ("B", Val::Float64(3.14159265))])
        .unwrap_err();

    assert!(
        err.to_string().contains("expected 3 value(s); got 2"),
        "{err}"
    );

    Ok(())
}

#[test]
fn variants() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let fragment = r#"
                (type $c' (record (field "D" bool) (field "E" u32)))
                (export $c "c" (type $c'))
                (type $Foo' (variant
                    (case "A" u32)
                    (case "B" float64)
                    (case "C" $c)
                ))
            "#;

    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            fragment,
            &[
                Param(Type::U8, Some(0)),
                Param(Type::I64, Some(8)),
                Param(Type::I32, None),
            ],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let input = ty
        .unwrap_variant()
        .new_val("B", Some(Val::Float64(3.14159265)))?;
    let mut output = [Val::Bool(false)];
    func.call_and_post_return(&mut store, &[input.clone()], &mut output)?;

    assert_eq!(input, output[0]);

    // Do it again, this time using case "C"

    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            fragment,
            &[
                Param(Type::U8, Some(0)),
                Param(Type::I64, Some(8)),
                Param(Type::I32, Some(12)),
            ],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let c_type = &ty.unwrap_variant().cases().nth(2).unwrap().ty.unwrap();
    let input = ty.unwrap_variant().new_val(
        "C",
        Some(
            c_type
                .unwrap_record()
                .new_val([("D", Val::Bool(true)), ("E", Val::U32(314159265))])?,
        ),
    )?;
    func.call_and_post_return(&mut store, &[input.clone()], &mut output)?;

    assert_eq!(input, output[0]);

    // Sad path: type mismatch

    let err = ty
        .unwrap_variant()
        .new_val("B", Some(Val::U64(314159265)))
        .unwrap_err();
    assert!(err.to_string().contains("type mismatch"), "{err}");
    let err = ty.unwrap_variant().new_val("B", None).unwrap_err();
    assert!(
        err.to_string().contains("expected a payload for case `B`"),
        "{err}"
    );

    // Sad path: unknown case

    let err = ty
        .unwrap_variant()
        .new_val("D", Some(Val::U64(314159265)))
        .unwrap_err();
    assert!(err.to_string().contains("unknown variant case"), "{err}");
    let err = ty.unwrap_variant().new_val("D", None).unwrap_err();
    assert!(err.to_string().contains("unknown variant case"), "{err}");

    // Make sure we lift variants which have cases of different sizes with the correct alignment

    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"
                (type $c' (record (field "D" bool) (field "E" u32)))
                (export $c "c" (type $c'))
                (type $a' (variant
                    (case "A" u32)
                    (case "B" float64)
                    (case "C" $c)
                ))
                (export $a "a" (type $a'))
                (type $Foo' (record
                    (field "A" $a)
                    (field "B" u32)
                ))
            "#,
            &[
                Param(Type::U8, Some(0)),
                Param(Type::I64, Some(8)),
                Param(Type::I32, None),
                Param(Type::I32, Some(16)),
            ],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let a_type = &ty.unwrap_record().fields().nth(0).unwrap().ty;
    let input = ty.unwrap_record().new_val([
        (
            "A",
            a_type
                .unwrap_variant()
                .new_val("A", Some(Val::U32(314159265)))?,
        ),
        ("B", Val::U32(628318530)),
    ])?;
    func.call_and_post_return(&mut store, &[input.clone()], &mut output)?;

    assert_eq!(input, output[0]);

    Ok(())
}

#[test]
fn flags() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"(flags "A" "B" "C" "D" "E")"#,
            &[Param(Type::U8, Some(0))],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let input = ty.unwrap_flags().new_val(&["B", "D"])?;
    let mut output = [Val::Bool(false)];
    func.call_and_post_return(&mut store, &[input.clone()], &mut output)?;

    assert_eq!(input, output[0]);

    // Sad path: unknown flags

    let err = ty.unwrap_flags().new_val(&["B", "D", "F"]).unwrap_err();

    assert!(err.to_string().contains("unknown flag"), "{err}");

    Ok(())
}

#[test]
fn everything() -> Result<()> {
    // This serves to test both nested types and storing parameters on the heap (i.e. exceeding `MAX_STACK_PARAMS`)

    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"
                (type $b' (enum "a" "b"))
                (export $b "b" (type $b'))
                (type $c' (record (field "D" bool) (field "E" u32)))
                (export $c "c" (type $c'))
                (type $f' (flags "G" "H" "I"))
                (export $f "f" (type $f'))
                (type $m' (record (field "N" bool) (field "O" u32)))
                (export $m "m" (type $m'))
                (type $j' (variant
                    (case "K" u32)
                    (case "L" float64)
                    (case "M" $m)
                ))
                (export $j "j" (type $j'))

                (type $Foo' (record
                    (field "A" u32)
                    (field "B" $b)
                    (field "C" $c)
                    (field "F" (list $f))
                    (field "J" $j)
                    (field "P" s8)
                    (field "Q" s16)
                    (field "R" s32)
                    (field "S" s64)
                    (field "T" float32)
                    (field "U" float64)
                    (field "V" string)
                    (field "W" char)
                    (field "Y" (tuple u32 u32))
                    (field "AA" (option u32))
                    (field "BB" (result string (error string)))
                ))
            "#,
            &[
                Param(Type::I32, Some(0)),
                Param(Type::U8, Some(4)),
                Param(Type::U8, Some(5)),
                Param(Type::I32, Some(8)),
                Param(Type::I32, Some(12)),
                Param(Type::I32, Some(16)),
                Param(Type::U8, Some(20)),
                Param(Type::I64, Some(28)),
                Param(Type::I32, Some(32)),
                Param(Type::S8, Some(36)),
                Param(Type::S16, Some(38)),
                Param(Type::I32, Some(40)),
                Param(Type::I64, Some(48)),
                Param(Type::F32, Some(56)),
                Param(Type::F64, Some(64)),
                Param(Type::I32, Some(72)),
                Param(Type::I32, Some(76)),
                Param(Type::I32, Some(80)),
                Param(Type::I32, Some(84)),
                Param(Type::I32, Some(88)),
                Param(Type::I64, Some(96)),
                Param(Type::U8, Some(104)),
                Param(Type::I32, Some(108)),
                Param(Type::U8, Some(112)),
                Param(Type::I32, Some(116)),
                Param(Type::I32, Some(120)),
            ],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let types = ty
        .unwrap_record()
        .fields()
        .map(|field| field.ty)
        .collect::<Box<[component::Type]>>();
    let (b_type, c_type, f_type, j_type, y_type, aa_type, bb_type) = (
        &types[1], &types[2], &types[3], &types[4], &types[13], &types[14], &types[15],
    );
    let f_element_type = &f_type.unwrap_list().ty();
    let input = ty.unwrap_record().new_val([
        ("A", Val::U32(32343)),
        ("B", b_type.unwrap_enum().new_val("b")?),
        (
            "C",
            c_type
                .unwrap_record()
                .new_val([("D", Val::Bool(false)), ("E", Val::U32(2084037802))])?,
        ),
        (
            "F",
            f_type.unwrap_list().new_val(Box::new([f_element_type
                .unwrap_flags()
                .new_val(&["G", "I"])?]))?,
        ),
        (
            "J",
            j_type
                .unwrap_variant()
                .new_val("L", Some(Val::Float64(3.14159265)))?,
        ),
        ("P", Val::S8(42)),
        ("Q", Val::S16(4242)),
        ("R", Val::S32(42424242)),
        ("S", Val::S64(424242424242424242)),
        ("T", Val::Float32(3.14159265)),
        ("U", Val::Float64(3.14159265)),
        ("V", Val::String(Box::from("wow, nice types"))),
        ("W", Val::Char('🦀')),
        (
            "Y",
            y_type
                .unwrap_tuple()
                .new_val(Box::new([Val::U32(42), Val::U32(24)]))?,
        ),
        (
            "AA",
            aa_type.unwrap_option().new_val(Some(Val::U32(314159265)))?,
        ),
        (
            "BB",
            bb_type
                .unwrap_result()
                .new_val(Ok(Some(Val::String(Box::from("no problem")))))?,
        ),
    ])?;
    let mut output = [Val::Bool(false)];
    func.call_and_post_return(&mut store, &[input.clone()], &mut output)?;

    assert_eq!(input, output[0]);

    Ok(())
}

#[test]
fn introspection() -> Result<()> {
    let engine = super::engine();

    let component = Component::new(
        &engine,
        format!(
            r#"
            (component
                (import "res" (type $res (sub resource)))

                (type $b' (enum "a" "b"))
                (export $b "b" (type $b'))
                (type $c' (record (field "D" bool) (field "E" u32)))
                (export $c "c" (type $c'))
                (type $f' (flags "G" "H" "I"))
                (export $f "f" (type $f'))
                (type $m' (record (field "N" bool) (field "O" u32)))
                (export $m "m" (type $m'))
                (type $j' (variant
                    (case "K" u32)
                    (case "L" float64)
                    (case "M" $m)
                ))
                (export $j "j" (type $j'))

                (type $Foo' (record
                    (field "A" u32)
                    (field "B" $b)
                    (field "C" $c)
                    (field "F" (list $f))
                    (field "J" $j)
                    (field "P" s8)
                    (field "Q" s16)
                    (field "R" s32)
                    (field "S" s64)
                    (field "T" float32)
                    (field "U" float64)
                    (field "V" string)
                    (field "W" char)
                    (field "Y" (tuple u32 u32))
                    (field "AA" (option u32))
                    (field "BB" (result string (error string)))
                    (field "CC" (own $res))
                ))
                (export $Foo "foo" (type $Foo'))

                (core module $m
                    (func (export "f") (param i32) (result i32)
                        local.get 0
                    )
                    (memory (export "memory") 1)
                    {REALLOC_AND_FREE}
                )
                (core instance $i (instantiate $m))

                (func (export "fn") (param "x" (option $Foo)) (result (option (tuple u32 u32)))
                    (canon lift
                        (core func $i "f")
                        (memory $i "memory")
                        (realloc (func $i "realloc"))
                    )
                )
            )
        "#
        ),
    )?;

    struct MyType;

    let mut linker = Linker::<()>::new(&engine);
    linker
        .root()
        .resource("res", ResourceType::host::<MyType>(), |_, _| Ok(()))?;

    let component_ty = linker.substituted_component_type(&component)?;

    let mut imports = component_ty.imports();
    assert_eq!(imports.len(), 1);
    let (name, res_ty) = imports.next().unwrap();
    assert_eq!(name, "res");
    let ComponentItem::Resource(res_ty) = res_ty else {
        panic!("`res` import item of wrong type")
    };
    assert_eq!(res_ty, ResourceType::host::<MyType>());

    let mut exports = component_ty.exports();
    assert_eq!(exports.len(), 7);
    let (name, b_ty) = exports.next().unwrap();
    assert_eq!(name, "b");
    let ComponentItem::Type(b_ty) = b_ty else {
        panic!("`b` export item of wrong type")
    };
    assert_eq!(b_ty.unwrap_enum().names().collect::<Vec<_>>(), ["a", "b"]);

    let (name, c_ty) = exports.next().unwrap();
    assert_eq!(name, "c");
    let ComponentItem::Type(c_ty) = c_ty else {
        panic!("`c` export item of wrong type")
    };
    let mut fields = c_ty.unwrap_record().fields();
    {
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "D");
        assert_eq!(ty, types::Type::Bool);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "E");
        assert_eq!(ty, types::Type::U32);
    }

    let (name, f_ty) = exports.next().unwrap();
    assert_eq!(name, "f");
    let ComponentItem::Type(f_ty) = f_ty else {
        panic!("`f` export item of wrong type")
    };
    assert_eq!(
        f_ty.unwrap_flags().names().collect::<Vec<_>>(),
        ["G", "H", "I"]
    );

    let (name, m_ty) = exports.next().unwrap();
    assert_eq!(name, "m");
    let ComponentItem::Type(m_ty) = m_ty else {
        panic!("`m` export item of wrong type")
    };
    {
        let mut fields = m_ty.unwrap_record().fields();
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "N");
        assert_eq!(ty, types::Type::Bool);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "O");
        assert_eq!(ty, types::Type::U32);
    }

    let (name, j_ty) = exports.next().unwrap();
    assert_eq!(name, "j");
    let ComponentItem::Type(j_ty) = j_ty else {
        panic!("`j` export item of wrong type")
    };
    let mut cases = j_ty.unwrap_variant().cases();
    {
        let Case { name, ty } = cases.next().unwrap();
        assert_eq!(name, "K");
        assert_eq!(ty, Some(types::Type::U32));
        let Case { name, ty } = cases.next().unwrap();
        assert_eq!(name, "L");
        assert_eq!(ty, Some(types::Type::Float64));
        let Case { name, ty } = cases.next().unwrap();
        assert_eq!(name, "M");
        assert_eq!(ty, Some(m_ty));
    }

    let (name, foo_ty) = exports.next().unwrap();
    assert_eq!(name, "foo");
    let ComponentItem::Type(foo_ty) = foo_ty else {
        panic!("`foo` export item of wrong type")
    };
    {
        let mut fields = foo_ty.unwrap_record().fields();
        assert_eq!(fields.len(), 17);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "A");
        assert_eq!(ty, types::Type::U32);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "B");
        assert_eq!(ty, b_ty);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "C");
        assert_eq!(ty, c_ty);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "F");
        let ty = ty.unwrap_list();
        assert_eq!(ty.ty(), f_ty);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "J");
        assert_eq!(ty, j_ty);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "P");
        assert_eq!(ty, types::Type::S8);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "Q");
        assert_eq!(ty, types::Type::S16);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "R");
        assert_eq!(ty, types::Type::S32);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "S");
        assert_eq!(ty, types::Type::S64);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "T");
        assert_eq!(ty, types::Type::Float32);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "U");
        assert_eq!(ty, types::Type::Float64);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "V");
        assert_eq!(ty, types::Type::String);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "W");
        assert_eq!(ty, types::Type::Char);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "Y");
        assert_eq!(
            ty.unwrap_tuple().types().collect::<Vec<_>>(),
            [types::Type::U32, types::Type::U32]
        );
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "AA");
        assert_eq!(ty.unwrap_option().ty(), types::Type::U32);
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "BB");
        let ty = ty.unwrap_result();
        assert_eq!(ty.ok(), Some(types::Type::String));
        assert_eq!(ty.err(), Some(types::Type::String));
        let Field { name, ty } = fields.next().unwrap();
        assert_eq!(name, "CC");
        assert_eq!(*ty.unwrap_own(), res_ty);
    }

    let (name, fn_ty) = exports.next().unwrap();
    assert_eq!(name, "fn");
    let ComponentItem::ComponentFunc(fn_ty) = fn_ty else {
        panic!("`fn` export item of wrong type")
    };
    let mut params = fn_ty.params();
    assert_eq!(params.len(), 1);
    assert_eq!(params.next().unwrap().unwrap_option().ty(), foo_ty);

    let mut results = fn_ty.results();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results
            .next()
            .unwrap()
            .unwrap_option()
            .ty()
            .unwrap_tuple()
            .types()
            .collect::<Vec<_>>(),
        [types::Type::U32, types::Type::U32]
    );
    Ok(())
}
