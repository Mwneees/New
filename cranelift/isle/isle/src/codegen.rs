//! Generate Rust code from a series of Sequences.

use crate::ir::{ExprInst, InstId, PatternInst, Value};
use crate::log;
use crate::sema::{ExternalSig, ReturnKind, TermEnv, TermId, Type, TypeEnv, TypeId, Variant};
use crate::trie::{TrieEdge, TrieNode, TrieSymbol};
use crate::{StableMap, StableSet};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write;

/// Options for code generation.
#[derive(Clone, Debug, Default)]
pub struct CodegenOptions {
    /// Do not include the `#![allow(...)]` pragmas in the generated
    /// source. Useful if it must be include!()'d elsewhere.
    pub exclude_global_allow_pragmas: bool,
}

/// Emit Rust source code for the given type and term environments.
pub fn codegen(
    typeenv: &TypeEnv,
    termenv: &TermEnv,
    tries: &BTreeMap<TermId, TrieNode>,
    options: &CodegenOptions,
) -> String {
    Codegen::compile(typeenv, termenv, tries).generate_rust(options)
}

#[derive(Clone, Debug)]
struct Codegen<'a> {
    typeenv: &'a TypeEnv,
    termenv: &'a TermEnv,
    functions_by_term: &'a BTreeMap<TermId, TrieNode>,
}

#[derive(Clone, Debug, Default)]
struct BodyContext {
    /// For each value: (is_ref, ty).
    values: StableMap<Value, (bool, TypeId)>,
}

impl<'a> Codegen<'a> {
    fn compile(
        typeenv: &'a TypeEnv,
        termenv: &'a TermEnv,
        tries: &'a BTreeMap<TermId, TrieNode>,
    ) -> Codegen<'a> {
        Codegen {
            typeenv,
            termenv,
            functions_by_term: tries,
        }
    }

    fn generate_rust(&self, options: &CodegenOptions) -> String {
        let mut code = String::new();

        self.generate_header(&mut code, options);
        self.generate_ctx_trait(&mut code);
        self.generate_internal_types(&mut code);
        self.generate_internal_term_constructors(&mut code);

        code
    }

    fn generate_header(&self, code: &mut String, options: &CodegenOptions) {
        writeln!(code, "// GENERATED BY ISLE. DO NOT EDIT!").unwrap();
        writeln!(code, "//").unwrap();
        writeln!(
            code,
            "// Generated automatically from the instruction-selection DSL code in:",
        )
        .unwrap();
        for file in &self.typeenv.filenames {
            writeln!(code, "// - {}", file).unwrap();
        }

        if !options.exclude_global_allow_pragmas {
            writeln!(
                code,
                "\n#![allow(dead_code, unreachable_code, unreachable_patterns)]"
            )
            .unwrap();
            writeln!(
                code,
                "#![allow(unused_imports, unused_variables, non_snake_case, unused_mut)]"
            )
            .unwrap();
            writeln!(
                code,
                "#![allow(irrefutable_let_patterns, unused_assignments, non_camel_case_types)]"
            )
            .unwrap();
        }

        writeln!(code, "\nuse super::*;  // Pulls in all external types.").unwrap();
        writeln!(code, "use std::marker::PhantomData;").unwrap();
    }

    fn generate_trait_sig(&self, code: &mut String, indent: &str, sig: &ExternalSig) {
        let ret_tuple = format!(
            "{open_paren}{rets}{close_paren}",
            open_paren = if sig.ret_tys.len() != 1 { "(" } else { "" },
            rets = sig
                .ret_tys
                .iter()
                .map(|&ty| self.type_name(ty, /* by_ref = */ false))
                .collect::<Vec<_>>()
                .join(", "),
            close_paren = if sig.ret_tys.len() != 1 { ")" } else { "" },
        );

        if sig.ret_kind == ReturnKind::Iterator {
            writeln!(
                code,
                "{indent}type {name}_iter: ContextIter<Context = Self, Output = {output}>;",
                indent = indent,
                name = sig.func_name,
                output = ret_tuple,
            )
            .unwrap();
        }

        let ret_ty = match sig.ret_kind {
            ReturnKind::Plain => ret_tuple,
            ReturnKind::Option => format!("Option<{}>", ret_tuple),
            ReturnKind::Iterator => format!("Self::{}_iter", sig.func_name),
        };

        writeln!(
            code,
            "{indent}fn {name}(&mut self, {params}) -> {ret_ty};",
            indent = indent,
            name = sig.func_name,
            params = sig
                .param_tys
                .iter()
                .enumerate()
                .map(|(i, &ty)| format!("arg{}: {}", i, self.type_name(ty, /* by_ref = */ true)))
                .collect::<Vec<_>>()
                .join(", "),
            ret_ty = ret_ty,
        )
        .unwrap();
    }

    fn generate_ctx_trait(&self, code: &mut String) {
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "/// Context during lowering: an implementation of this trait"
        )
        .unwrap();
        writeln!(
            code,
            "/// must be provided with all external constructors and extractors."
        )
        .unwrap();
        writeln!(
            code,
            "/// A mutable borrow is passed along through all lowering logic."
        )
        .unwrap();
        writeln!(code, "pub trait Context {{").unwrap();
        for term in &self.termenv.terms {
            if term.has_external_extractor() {
                let ext_sig = term.extractor_sig(self.typeenv).unwrap();
                self.generate_trait_sig(code, "    ", &ext_sig);
            }
            if term.has_external_constructor() {
                let ext_sig = term.constructor_sig(self.typeenv).unwrap();
                self.generate_trait_sig(code, "    ", &ext_sig);
            }
        }
        writeln!(code, "}}").unwrap();
        writeln!(
            code,
            r#"
           pub trait ContextIter {{
               type Context;
               type Output;
               fn next(&mut self, ctx: &mut Self::Context) -> Option<Self::Output>;
           }}

           pub struct ContextIterWrapper<Item, I: Iterator < Item = Item>, C: Context> {{
               iter: I,
               _ctx: PhantomData<C>,
           }}
           impl<Item, I: Iterator<Item = Item>, C: Context> From<I> for ContextIterWrapper<Item, I, C> {{
               fn from(iter: I) -> Self {{
                   Self {{ iter, _ctx: PhantomData }}
               }}
           }}
           impl<Item, I: Iterator<Item = Item>, C: Context> ContextIter for ContextIterWrapper<Item, I, C> {{
               type Context = C;
               type Output = Item;
               fn next(&mut self, _ctx: &mut Self::Context) -> Option<Self::Output> {{
                   self.iter.next()
               }}
           }}
           "#,
        )
            .unwrap();
    }

    fn generate_internal_types(&self, code: &mut String) {
        for ty in &self.typeenv.types {
            match ty {
                &Type::Enum {
                    name,
                    is_extern,
                    is_nodebug,
                    ref variants,
                    pos,
                    ..
                } if !is_extern => {
                    let name = &self.typeenv.syms[name.index()];
                    writeln!(
                        code,
                        "\n/// Internal type {}: defined at {}.",
                        name,
                        pos.pretty_print_line(&self.typeenv.filenames[..])
                    )
                    .unwrap();

                    // Generate the `derive`s.
                    let debug_derive = if is_nodebug { "" } else { ", Debug" };
                    if variants.iter().all(|v| v.fields.is_empty()) {
                        writeln!(
                            code,
                            "#[derive(Copy, Clone, PartialEq, Eq{})]",
                            debug_derive
                        )
                        .unwrap();
                    } else {
                        writeln!(code, "#[derive(Clone{})]", debug_derive).unwrap();
                    }

                    writeln!(code, "pub enum {} {{", name).unwrap();
                    for variant in variants {
                        let name = &self.typeenv.syms[variant.name.index()];
                        if variant.fields.is_empty() {
                            writeln!(code, "    {},", name).unwrap();
                        } else {
                            writeln!(code, "    {} {{", name).unwrap();
                            for field in &variant.fields {
                                let name = &self.typeenv.syms[field.name.index()];
                                let ty_name =
                                    self.typeenv.types[field.ty.index()].name(&self.typeenv);
                                writeln!(code, "        {}: {},", name, ty_name).unwrap();
                            }
                            writeln!(code, "    }},").unwrap();
                        }
                    }
                    writeln!(code, "}}").unwrap();
                }
                _ => {}
            }
        }
    }

    fn type_name(&self, typeid: TypeId, by_ref: bool) -> String {
        match &self.typeenv.types[typeid.index()] {
            &Type::Primitive(_, sym, _) => self.typeenv.syms[sym.index()].clone(),
            &Type::Enum { name, .. } => {
                let r = if by_ref { "&" } else { "" };
                format!("{}{}", r, self.typeenv.syms[name.index()])
            }
        }
    }

    fn value_name(&self, value: &Value) -> String {
        match value {
            &Value::Pattern { inst, output } => format!("pattern{}_{}", inst.index(), output),
            &Value::Expr { inst, output } => format!("expr{}_{}", inst.index(), output),
        }
    }

    fn ty_prim(&self, ty: TypeId) -> bool {
        self.typeenv.types[ty.index()].is_prim()
    }

    fn value_binder(&self, value: &Value, is_ref: bool, ty: TypeId) -> String {
        let prim = self.ty_prim(ty);
        if prim || !is_ref {
            format!("{}", self.value_name(value))
        } else {
            format!("ref {}", self.value_name(value))
        }
    }

    fn value_by_ref(&self, value: &Value, ctx: &BodyContext) -> String {
        let raw_name = self.value_name(value);
        let &(is_ref, ty) = ctx.values.get(value).unwrap();
        let prim = self.ty_prim(ty);
        if is_ref || prim {
            raw_name
        } else {
            format!("&{}", raw_name)
        }
    }

    fn value_by_val(&self, value: &Value, ctx: &BodyContext) -> String {
        let raw_name = self.value_name(value);
        let &(is_ref, _) = ctx.values.get(value).unwrap();
        if is_ref {
            format!("{}.clone()", raw_name)
        } else {
            raw_name
        }
    }

    fn define_val(&self, value: &Value, ctx: &mut BodyContext, is_ref: bool, ty: TypeId) {
        let is_ref = !self.ty_prim(ty) && is_ref;
        ctx.values.insert(value.clone(), (is_ref, ty));
    }

    fn const_int(&self, val: i128, ty: TypeId) -> String {
        let is_bool = match &self.typeenv.types[ty.index()] {
            &Type::Primitive(_, name, _) => &self.typeenv.syms[name.index()] == "bool",
            _ => unreachable!(),
        };
        if is_bool {
            format!("{}", val != 0)
        } else {
            let ty_name = self.type_name(ty, /* by_ref = */ false);
            if ty_name == "i128" {
                format!("{}i128", val)
            } else {
                format!("{}i128 as {}", val, ty_name)
            }
        }
    }

    fn generate_internal_term_constructors(&self, code: &mut String) {
        for (&termid, trie) in self.functions_by_term {
            let termdata = &self.termenv.terms[termid.index()];

            // Skip terms that are enum variants or that have external
            // constructors/extractors.
            if !termdata.has_constructor() || termdata.has_external_constructor() {
                continue;
            }

            let sig = termdata.constructor_sig(self.typeenv).unwrap();

            let args = sig
                .param_tys
                .iter()
                .enumerate()
                .map(|(i, &ty)| format!("arg{}: {}", i, self.type_name(ty, true)))
                .collect::<Vec<_>>()
                .join(", ");
            assert_eq!(sig.ret_tys.len(), 1);

            let ret = self.type_name(sig.ret_tys[0], false);
            let ret = match sig.ret_kind {
                ReturnKind::Iterator => format!("impl ContextIter<Context = C, Output = {}>", ret),
                ReturnKind::Option => format!("Option<{}>", ret),
                ReturnKind::Plain => ret,
            };

            let term_name = &self.typeenv.syms[termdata.name.index()];
            writeln!(
                code,
                "\n// Generated as internal constructor for term {}.",
                term_name,
            )
            .unwrap();
            writeln!(
                code,
                "pub fn {}<C: Context>(ctx: &mut C, {}) -> {} {{",
                sig.func_name, args, ret,
            )
            .unwrap();

            if sig.ret_kind == ReturnKind::Iterator {
                writeln!(code, "let mut returns = ConstructorVec::new();").unwrap();
            }

            let mut body_ctx: BodyContext = Default::default();
            let returned = self.generate_body(
                code,
                /* depth = */ 0,
                trie,
                "    ",
                &mut body_ctx,
                sig.ret_kind,
            );
            if !returned {
                let ret_expr = match sig.ret_kind {
                    ReturnKind::Plain => Cow::from(format!(
                        "unreachable!(\"no rule matched for term {{}} at {{}}; should it be partial?\", {:?}, {:?})",
                        term_name,
                        termdata
                            .decl_pos
                            .pretty_print_line(&self.typeenv.filenames[..])
                    )),
                    ReturnKind::Option => Cow::from("None"),
                    ReturnKind::Iterator => {
                        Cow::from("ContextIterWrapper::from(returns.into_iter())")
                    }
                };
                write!(code, "    return {};", ret_expr).unwrap();
            }

            writeln!(code, "}}").unwrap();
        }
    }

    fn generate_expr_inst(
        &self,
        code: &mut String,
        id: InstId,
        inst: &ExprInst,
        indent: &str,
        ctx: &mut BodyContext,
        returns: &mut Vec<(usize, String)>,
    ) -> bool {
        log!("generate_expr_inst: {:?}", inst);
        let mut new_scope = false;
        match inst {
            &ExprInst::ConstInt { ty, val } => {
                let value = Value::Expr {
                    inst: id,
                    output: 0,
                };
                self.define_val(&value, ctx, /* is_ref = */ false, ty);
                let name = self.value_name(&value);
                let ty_name = self.type_name(ty, /* by_ref = */ false);
                writeln!(
                    code,
                    "{}let {}: {} = {};",
                    indent,
                    name,
                    ty_name,
                    self.const_int(val, ty)
                )
                .unwrap();
            }
            &ExprInst::ConstPrim { ty, val } => {
                let value = Value::Expr {
                    inst: id,
                    output: 0,
                };
                self.define_val(&value, ctx, /* is_ref = */ false, ty);
                let name = self.value_name(&value);
                let ty_name = self.type_name(ty, /* by_ref = */ false);
                writeln!(
                    code,
                    "{}let {}: {} = {};",
                    indent,
                    name,
                    ty_name,
                    self.typeenv.syms[val.index()],
                )
                .unwrap();
            }
            &ExprInst::CreateVariant {
                ref inputs,
                ty,
                variant,
            } => {
                let variantinfo = match &self.typeenv.types[ty.index()] {
                    &Type::Primitive(..) => panic!("CreateVariant with primitive type"),
                    &Type::Enum { ref variants, .. } => &variants[variant.index()],
                };
                let mut input_fields = vec![];
                for ((input_value, _), field) in inputs.iter().zip(variantinfo.fields.iter()) {
                    let field_name = &self.typeenv.syms[field.name.index()];
                    let value_expr = self.value_by_val(input_value, ctx);
                    input_fields.push(format!("{}: {}", field_name, value_expr));
                }

                let output = Value::Expr {
                    inst: id,
                    output: 0,
                };
                let outputname = self.value_name(&output);
                let full_variant_name = format!(
                    "{}::{}",
                    self.type_name(ty, false),
                    self.typeenv.syms[variantinfo.name.index()]
                );
                if input_fields.is_empty() {
                    writeln!(
                        code,
                        "{}let {} = {};",
                        indent, outputname, full_variant_name
                    )
                    .unwrap();
                } else {
                    writeln!(
                        code,
                        "{}let {} = {} {{",
                        indent, outputname, full_variant_name
                    )
                    .unwrap();
                    for input_field in input_fields {
                        writeln!(code, "{}    {},", indent, input_field).unwrap();
                    }
                    writeln!(code, "{}}};", indent).unwrap();
                }
                self.define_val(&output, ctx, /* is_ref = */ false, ty);
            }
            &ExprInst::Construct {
                ref inputs,
                term,
                infallible,
                multi,
                ..
            } => {
                let mut input_exprs = vec![];
                for (input_value, input_ty) in inputs {
                    let value_expr = if self.typeenv.types[input_ty.index()].is_prim() {
                        self.value_by_val(input_value, ctx)
                    } else {
                        self.value_by_ref(input_value, ctx)
                    };
                    input_exprs.push(value_expr);
                }

                let output = Value::Expr {
                    inst: id,
                    output: 0,
                };
                let outputname = self.value_name(&output);
                let termdata = &self.termenv.terms[term.index()];
                let sig = termdata.constructor_sig(self.typeenv).unwrap();
                assert_eq!(input_exprs.len(), sig.param_tys.len());

                if !multi {
                    let fallible_try = if infallible { "" } else { "?" };
                    writeln!(
                        code,
                        "{}let {} = {}(ctx, {}){};",
                        indent,
                        outputname,
                        sig.full_name,
                        input_exprs.join(", "),
                        fallible_try,
                    )
                    .unwrap();
                } else {
                    writeln!(
                        code,
                        "{}let mut iter = {}(ctx, {});",
                        indent,
                        sig.full_name,
                        input_exprs.join(", "),
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "{}while let Some({}) = iter.next(ctx) {{",
                        indent, outputname,
                    )
                    .unwrap();
                    new_scope = true;
                }
                self.define_val(&output, ctx, /* is_ref = */ false, termdata.ret_ty);
            }
            &ExprInst::Return {
                index, ref value, ..
            } => {
                let value_expr = self.value_by_val(value, ctx);
                returns.push((index, value_expr));
            }
        }

        new_scope
    }

    fn match_variant_binders(
        &self,
        variant: &Variant,
        arg_tys: &[TypeId],
        id: InstId,
        ctx: &mut BodyContext,
    ) -> Vec<String> {
        arg_tys
            .iter()
            .zip(variant.fields.iter())
            .enumerate()
            .map(|(i, (&ty, field))| {
                let value = Value::Pattern {
                    inst: id,
                    output: i,
                };
                let valuename = self.value_binder(&value, /* is_ref = */ true, ty);
                let fieldname = &self.typeenv.syms[field.name.index()];
                self.define_val(&value, ctx, /* is_ref = */ true, field.ty);
                format!("{}: {}", fieldname, valuename)
            })
            .collect::<Vec<_>>()
    }

    /// Returns a `bool` indicating whether this pattern inst is
    /// infallible, and the number of scopes opened.
    fn generate_pattern_inst(
        &self,
        code: &mut String,
        id: InstId,
        inst: &PatternInst,
        indent: &str,
        ctx: &mut BodyContext,
    ) -> (bool, usize) {
        match inst {
            &PatternInst::Arg { index, ty } => {
                let output = Value::Pattern {
                    inst: id,
                    output: 0,
                };
                let outputname = self.value_name(&output);
                let is_ref = match &self.typeenv.types[ty.index()] {
                    &Type::Primitive(..) => false,
                    _ => true,
                };
                writeln!(code, "{}let {} = arg{};", indent, outputname, index).unwrap();
                self.define_val(
                    &Value::Pattern {
                        inst: id,
                        output: 0,
                    },
                    ctx,
                    is_ref,
                    ty,
                );
                (true, 0)
            }
            &PatternInst::MatchEqual { ref a, ref b, .. } => {
                let a = self.value_by_ref(a, ctx);
                let b = self.value_by_ref(b, ctx);
                writeln!(code, "{}if {} == {} {{", indent, a, b).unwrap();
                (false, 1)
            }
            &PatternInst::MatchInt {
                ref input,
                int_val,
                ty,
                ..
            } => {
                let int_val = self.const_int(int_val, ty);
                let input = self.value_by_val(input, ctx);
                writeln!(code, "{}if {} == {}  {{", indent, input, int_val).unwrap();
                (false, 1)
            }
            &PatternInst::MatchPrim { ref input, val, .. } => {
                let input = self.value_by_val(input, ctx);
                let sym = &self.typeenv.syms[val.index()];
                writeln!(code, "{}if {} == {} {{", indent, input, sym).unwrap();
                (false, 1)
            }
            &PatternInst::MatchVariant {
                ref input,
                input_ty,
                variant,
                ref arg_tys,
            } => {
                let input = self.value_by_ref(input, ctx);
                let variants = match &self.typeenv.types[input_ty.index()] {
                    &Type::Primitive(..) => panic!("primitive type input to MatchVariant"),
                    &Type::Enum { ref variants, .. } => variants,
                };
                let ty_name = self.type_name(input_ty, /* is_ref = */ true);
                let variant = &variants[variant.index()];
                let variantname = &self.typeenv.syms[variant.name.index()];
                let args = self.match_variant_binders(variant, &arg_tys[..], id, ctx);
                let args = if args.is_empty() {
                    "".to_string()
                } else {
                    format!("{{ {} }}", args.join(", "))
                };
                writeln!(
                    code,
                    "{}if let {}::{} {} = {} {{",
                    indent, ty_name, variantname, args, input
                )
                .unwrap();
                (false, 1)
            }
            &PatternInst::Extract {
                ref inputs,
                ref output_tys,
                term,
                infallible,
                multi,
                ..
            } => {
                let termdata = &self.termenv.terms[term.index()];
                let sig = termdata.extractor_sig(self.typeenv).unwrap();

                let input_values = inputs
                    .iter()
                    .map(|input| self.value_by_ref(input, ctx))
                    .collect::<Vec<_>>();
                let output_binders = output_tys
                    .iter()
                    .enumerate()
                    .map(|(i, &ty)| {
                        let output_val = Value::Pattern {
                            inst: id,
                            output: i,
                        };
                        self.define_val(&output_val, ctx, /* is_ref = */ false, ty);
                        self.value_binder(&output_val, /* is_ref = */ false, ty)
                    })
                    .collect::<Vec<_>>();

                let bind_pattern = format!(
                    "{open_paren}{vars}{close_paren}",
                    open_paren = if output_binders.len() == 1 { "" } else { "(" },
                    vars = output_binders.join(", "),
                    close_paren = if output_binders.len() == 1 { "" } else { ")" }
                );
                let etor_call = format!(
                    "{name}(ctx, {args})",
                    name = sig.full_name,
                    args = input_values.join(", ")
                );

                if multi {
                    writeln!(code, "{indent}let mut iter = {etor_call};").unwrap();
                    writeln!(
                        code,
                        "{indent}while let Some({bind_pattern}) = iter.next(ctx) {{",
                    )
                    .unwrap();
                    (false, 1)
                } else if infallible {
                    writeln!(code, "{indent}let {bind_pattern} = {etor_call};").unwrap();
                    (true, 0)
                } else {
                    writeln!(code, "{indent}if let Some({bind_pattern}) = {etor_call} {{").unwrap();
                    (false, 1)
                }
            }
            &PatternInst::Expr {
                ref seq, output_ty, ..
            } if seq.is_const_int().is_some() => {
                let (ty, val) = seq.is_const_int().unwrap();
                assert_eq!(ty, output_ty);

                let output = Value::Pattern {
                    inst: id,
                    output: 0,
                };
                writeln!(
                    code,
                    "{}let {} = {};",
                    indent,
                    self.value_name(&output),
                    self.const_int(val, ty),
                )
                .unwrap();
                self.define_val(&output, ctx, /* is_ref = */ false, ty);
                (true, 0)
            }
            &PatternInst::Expr {
                ref seq, output_ty, ..
            } => {
                let closure_name = format!("closure{}", id.index());
                writeln!(code, "{}let mut {} = || {{", indent, closure_name).unwrap();
                let subindent = format!("{}    ", indent);
                let mut subctx = ctx.clone();
                let mut returns = vec![];
                for (id, inst) in seq.insts.iter().enumerate() {
                    let id = InstId(id);
                    let new_scope = self.generate_expr_inst(
                        code,
                        id,
                        inst,
                        &subindent,
                        &mut subctx,
                        &mut returns,
                    );
                    assert!(!new_scope);
                }
                assert_eq!(returns.len(), 1);
                writeln!(code, "{}return Some({});", subindent, returns[0].1).unwrap();
                writeln!(code, "{}}};", indent).unwrap();

                let output = Value::Pattern {
                    inst: id,
                    output: 0,
                };
                writeln!(
                    code,
                    "{}if let Some({}) = {}() {{",
                    indent,
                    self.value_binder(&output, /* is_ref = */ false, output_ty),
                    closure_name
                )
                .unwrap();
                self.define_val(&output, ctx, /* is_ref = */ false, output_ty);

                (false, 1)
            }
        }
    }

    fn generate_body(
        &self,
        code: &mut String,
        depth: usize,
        trie: &TrieNode,
        indent: &str,
        ctx: &mut BodyContext,
        ret_kind: ReturnKind,
    ) -> bool {
        log!("generate_body:\n{}", trie.pretty());
        let mut returned = false;
        match trie {
            &TrieNode::Empty => {}

            &TrieNode::Leaf { ref output, .. } => {
                writeln!(
                    code,
                    "{}// Rule at {}.",
                    indent,
                    output.pos.pretty_print_line(&self.typeenv.filenames[..])
                )
                .unwrap();

                // If this is a leaf node, generate the ExprSequence and return.
                let mut returns = vec![];
                let mut scopes = 0;
                let mut indent = indent.to_string();
                let orig_indent = indent.clone();
                for (id, inst) in output.insts.iter().enumerate() {
                    let id = InstId(id);
                    let new_scope =
                        self.generate_expr_inst(code, id, inst, &indent[..], ctx, &mut returns);
                    if new_scope {
                        scopes += 1;
                        indent.push_str("    ");
                    }
                }

                assert_eq!(returns.len(), 1);
                let (before, after) = match ret_kind {
                    ReturnKind::Plain => ("return ", ""),
                    ReturnKind::Option => ("return Some(", ")"),
                    ReturnKind::Iterator => ("returns.push(", ")"),
                };
                writeln!(code, "{}{}{}{};", indent, before, returns[0].1, after).unwrap();

                for _ in 0..scopes {
                    writeln!(code, "{}}}", orig_indent).unwrap();
                }

                returned = ret_kind != ReturnKind::Iterator;
            }

            &TrieNode::Decision { ref edges } => {
                // If this is a decision node, generate each match op
                // in turn (in priority order). Gather together
                // adjacent MatchVariant ops with the same input and
                // disjoint variants in order to create a `match`
                // rather than a chain of if-lets.

                let mut i = 0;
                while i < edges.len() {
                    // Gather adjacent match variants so that we can turn these
                    // into a `match` rather than a sequence of `if let`s.
                    let mut last = i;
                    let mut adjacent_variants = StableSet::new();
                    let mut adjacent_variant_input = None;
                    log!(
                        "edge: prio = {:?}, symbol = {:?}",
                        edges[i].prio,
                        edges[i].symbol
                    );
                    while last < edges.len() {
                        match &edges[last].symbol {
                            &TrieSymbol::Match {
                                op: PatternInst::MatchVariant { input, variant, .. },
                            } => {
                                if adjacent_variant_input.is_none() {
                                    adjacent_variant_input = Some(input);
                                }
                                if adjacent_variant_input == Some(input)
                                    && !adjacent_variants.contains(&variant)
                                {
                                    adjacent_variants.insert(variant);
                                    last += 1;
                                } else {
                                    break;
                                }
                            }
                            _ => {
                                break;
                            }
                        }
                    }

                    // Now `edges[i..last]` is a run of adjacent `MatchVariants`
                    // (possibly an empty one). Only use a `match` form if there
                    // are at least two adjacent options.
                    if last - i > 1 {
                        self.generate_body_matches(
                            code,
                            depth,
                            &edges[i..last],
                            indent,
                            ctx,
                            ret_kind,
                        );
                        i = last;
                        continue;
                    } else {
                        let &TrieEdge {
                            ref symbol,
                            ref node,
                            ..
                        } = &edges[i];
                        i += 1;

                        match symbol {
                            &TrieSymbol::EndOfMatch => {
                                returned = self.generate_body(
                                    code,
                                    depth + 1,
                                    node,
                                    indent,
                                    ctx,
                                    ret_kind,
                                );
                            }
                            &TrieSymbol::Match { ref op } => {
                                let id = InstId(depth);
                                let (infallible, new_scopes) =
                                    self.generate_pattern_inst(code, id, op, indent, ctx);
                                let mut subindent = indent.to_string();
                                for _ in 0..new_scopes {
                                    subindent.push_str("    ");
                                }
                                let sub_returned = self.generate_body(
                                    code,
                                    depth + 1,
                                    node,
                                    &subindent[..],
                                    ctx,
                                    ret_kind,
                                );
                                for _ in 0..new_scopes {
                                    writeln!(code, "{}}}", indent).unwrap();
                                }
                                if infallible && sub_returned {
                                    returned = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        returned
    }

    fn generate_body_matches(
        &self,
        code: &mut String,
        depth: usize,
        edges: &[TrieEdge],
        indent: &str,
        ctx: &mut BodyContext,
        ret_kind: ReturnKind,
    ) {
        let (input, input_ty) = match &edges[0].symbol {
            &TrieSymbol::Match {
                op:
                    PatternInst::MatchVariant {
                        input, input_ty, ..
                    },
            } => (input, input_ty),
            _ => unreachable!(),
        };
        let (input_ty_sym, variants) = match &self.typeenv.types[input_ty.index()] {
            &Type::Enum {
                ref name,
                ref variants,
                ..
            } => (name, variants),
            _ => unreachable!(),
        };
        let input_ty_name = &self.typeenv.syms[input_ty_sym.index()];

        // Emit the `match`.
        writeln!(
            code,
            "{}match {} {{",
            indent,
            self.value_by_ref(&input, ctx)
        )
        .unwrap();

        // Emit each case.
        for &TrieEdge {
            ref symbol,
            ref node,
            ..
        } in edges
        {
            let id = InstId(depth);
            let (variant, arg_tys) = match symbol {
                &TrieSymbol::Match {
                    op:
                        PatternInst::MatchVariant {
                            variant,
                            ref arg_tys,
                            ..
                        },
                } => (variant, arg_tys),
                _ => unreachable!(),
            };

            let variantinfo = &variants[variant.index()];
            let variantname = &self.typeenv.syms[variantinfo.name.index()];
            let fields = self.match_variant_binders(variantinfo, arg_tys, id, ctx);
            let fields = if fields.is_empty() {
                "".to_string()
            } else {
                format!("{{ {} }}", fields.join(", "))
            };
            writeln!(
                code,
                "{}    &{}::{} {} => {{",
                indent, input_ty_name, variantname, fields,
            )
            .unwrap();
            let subindent = format!("{}        ", indent);
            self.generate_body(code, depth + 1, node, &subindent, ctx, ret_kind);
            writeln!(code, "{}    }}", indent).unwrap();
        }

        // Always add a catchall, because we don't do exhaustiveness
        // checking on the MatchVariants.
        writeln!(code, "{}    _ => {{}}", indent).unwrap();

        writeln!(code, "{}}}", indent).unwrap();
    }
}
