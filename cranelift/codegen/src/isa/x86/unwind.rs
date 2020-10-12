//! Module for x86 unwind generation for supported ABIs.

pub mod systemv;
pub mod winx64;

use crate::ir::{Function, InstructionData, Opcode, ValueLoc};
use crate::isa::x86::registers::{FPR, RU};
use crate::isa::{RegUnit, TargetIsa};
use crate::result::CodegenResult;
use alloc::vec::Vec;

use crate::isa::unwind::input::{UnwindCode, UnwindInfo};

pub(crate) fn create_unwind_info(
    func: &Function,
    isa: &dyn TargetIsa,
    frame_register: Option<RegUnit>,
) -> CodegenResult<Option<UnwindInfo<RegUnit>>> {
    let mut len = 0;

    let entry_block = func.layout.entry_block().expect("missing entry block");
    let prologue_end = func.prologue_end.unwrap();

    let mut stack_size = None;
    let mut prologue_size = 0;
    let mut prologue_unwind_codes = Vec::new();
    let mut epilogues_unwind_codes = Vec::new();

    let mut blocks = func.layout.blocks().collect::<Vec<_>>();
    blocks.sort_by_key(|b| func.offsets[*b]);

    for (i, block) in blocks.iter().enumerate() {
        let mut in_prologue = block == &entry_block;
        let mut in_epilogue = false;
        let mut epilogue_pop_offsets = Vec::new();

        for (offset, inst, size) in func.inst_offsets(*block, &isa.encoding_info()) {
            let offset = offset + size;
            assert!(len <= offset);
            len = offset;

            let is_last_block = i == blocks.len() - 1;

            let unwind_codes;
            if in_prologue {
                // Check for prologue end (inclusive)
                if prologue_end == inst {
                    in_prologue = false;
                }
                prologue_size += size;
                unwind_codes = &mut prologue_unwind_codes;
            } else if !in_epilogue && func.epilogues_start.contains(&inst) {
                // Now in an epilogue, emit a remember state instruction if not last block
                in_epilogue = true;

                epilogues_unwind_codes.push(Vec::new());
                unwind_codes = epilogues_unwind_codes.last_mut().unwrap();

                if !is_last_block {
                    unwind_codes.push(UnwindCode::RememberState { offset });
                }
            } else if in_epilogue {
                unwind_codes = epilogues_unwind_codes.last_mut().unwrap();
            } else {
                // Ignore normal instructions
                continue;
            }

            match func.dfg[inst] {
                InstructionData::Unary { opcode, arg } => {
                    match opcode {
                        Opcode::X86Push => {
                            let reg = func.locations[arg].unwrap_reg();
                            unwind_codes.push(UnwindCode::PushRegister { offset, reg });
                        }
                        Opcode::AdjustSpDown => {
                            let stack_size =
                                stack_size.expect("expected a previous stack size instruction");

                            // This is used when calling a stack check function
                            // We need to track the assignment to RAX which has the size of the stack
                            unwind_codes.push(UnwindCode::StackAlloc {
                                offset,
                                size: stack_size,
                            });
                        }
                        _ => {}
                    }
                }
                InstructionData::UnaryImm { opcode, imm } => {
                    match opcode {
                        Opcode::Iconst => {
                            let imm: i64 = imm.into();
                            assert!(imm <= core::u32::MAX as i64);
                            assert!(stack_size.is_none());

                            // This instruction should only appear in a prologue to pass an
                            // argument of the stack size to a stack check function.
                            // Record the stack size so we know what it is when we encounter the adjustment
                            // instruction (which will adjust via the register assigned to this instruction).
                            stack_size = Some(imm as u32);
                        }
                        Opcode::AdjustSpDownImm => {
                            let imm: i64 = imm.into();
                            assert!(imm <= core::u32::MAX as i64);

                            stack_size = Some(imm as u32);

                            unwind_codes.push(UnwindCode::StackAlloc {
                                offset,
                                size: imm as u32,
                            });
                        }
                        Opcode::AdjustSpUpImm => {
                            let imm: i64 = imm.into();
                            assert!(imm <= core::u32::MAX as i64);

                            stack_size = Some(imm as u32);

                            unwind_codes.push(UnwindCode::StackDealloc {
                                offset,
                                size: imm as u32,
                            });
                        }
                        _ => {}
                    }
                }
                InstructionData::Store {
                    opcode: Opcode::Store,
                    args: [arg1, arg2],
                    offset: stack_offset,
                    ..
                } => {
                    if let (ValueLoc::Reg(src), ValueLoc::Reg(dst)) =
                        (func.locations[arg1], func.locations[arg2])
                    {
                        // If this is a save of an FPR, record an unwind operation
                        // Note: the stack_offset here is relative to an adjusted SP
                        if dst == (RU::rsp as RegUnit) && FPR.contains(src) {
                            let stack_offset: i32 = stack_offset.into();
                            unwind_codes.push(UnwindCode::SaveXmm {
                                offset,
                                reg: src,
                                stack_offset: stack_offset as u32,
                            });
                        }
                    }
                }
                InstructionData::CopySpecial { src, dst, .. } => {
                    if let Some(fp) = frame_register {
                        // Check for change in CFA register (RSP is always the starting CFA)
                        if src == (RU::rsp as RegUnit) && dst == fp {
                            unwind_codes.push(UnwindCode::SetCfaRegister { offset, reg: dst });
                        }
                    }
                }
                InstructionData::NullAry { opcode } => match opcode {
                    Opcode::X86Pop => {
                        epilogue_pop_offsets.push(offset);
                    }
                    _ => {}
                },
                InstructionData::MultiAry { opcode, .. } if in_epilogue => match opcode {
                    Opcode::Return => {
                        let args = func.dfg.inst_args(inst);
                        for (i, arg) in args.iter().rev().enumerate() {
                            // Only walk back the args for the pop instructions encountered
                            if i >= epilogue_pop_offsets.len() {
                                break;
                            }

                            let offset = epilogue_pop_offsets[i];

                            let reg = func.locations[*arg].unwrap_reg();
                            unwind_codes.push(UnwindCode::PopRegister { offset, reg });
                        }
                        epilogue_pop_offsets.clear();

                        if !is_last_block {
                            unwind_codes.push(UnwindCode::RestoreState { offset });
                        }

                        in_epilogue = false;
                    }
                    _ => {}
                },
                _ => {}
            };
        }
    }

    Ok(Some(UnwindInfo {
        prologue_size,
        prologue_unwind_codes,
        epilogues_unwind_codes,
        code_len: len,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::{Cursor, FuncCursor};
    use crate::ir::{
        types, AbiParam, ExternalName, InstBuilder, Signature, StackSlotData, StackSlotKind,
    };
    use crate::isa::{lookup, CallConv};
    use crate::settings::{builder, Flags};
    use crate::Context;
    use std::str::FromStr;
    use target_lexicon::triple;

    #[test]
    #[cfg_attr(feature = "x64", should_panic)] // TODO #2079
    fn test_small_alloc() {
        let isa = lookup(triple!("x86_64"))
            .expect("expect x86 ISA")
            .finish(Flags::new(builder()));

        let mut context = Context::for_function(create_function(
            CallConv::WindowsFastcall,
            Some(StackSlotData::new(StackSlotKind::ExplicitSlot, 64)),
        ));

        context.compile(&*isa).expect("expected compilation");

        let unwind = create_unwind_info(&context.func, &*isa, None)
            .expect("can create unwind info")
            .expect("expected unwind info");

        assert_eq!(
            unwind,
            UnwindInfo {
                prologue_size: 9,
                prologue_unwind_codes: vec![
                    UnwindCode::PushRegister {
                        offset: 2,
                        reg: RU::rbp.into(),
                    },
                    UnwindCode::StackAlloc {
                        offset: 9,
                        size: 64
                    }
                ],
                epilogues_unwind_codes: vec![vec![
                    UnwindCode::StackDealloc {
                        offset: 13,
                        size: 64
                    },
                    UnwindCode::PopRegister {
                        offset: 15,
                        reg: RU::rbp.into()
                    }
                ]],
                code_len: 16,
            }
        );
    }

    #[test]
    #[cfg_attr(feature = "x64", should_panic)] // TODO #2079
    fn test_medium_alloc() {
        let isa = lookup(triple!("x86_64"))
            .expect("expect x86 ISA")
            .finish(Flags::new(builder()));

        let mut context = Context::for_function(create_function(
            CallConv::WindowsFastcall,
            Some(StackSlotData::new(StackSlotKind::ExplicitSlot, 10000)),
        ));

        context.compile(&*isa).expect("expected compilation");

        let unwind = create_unwind_info(&context.func, &*isa, None)
            .expect("can create unwind info")
            .expect("expected unwind info");

        assert_eq!(
            unwind,
            UnwindInfo {
                prologue_size: 27,
                prologue_unwind_codes: vec![
                    UnwindCode::PushRegister {
                        offset: 2,
                        reg: RU::rbp.into(),
                    },
                    UnwindCode::StackAlloc {
                        offset: 27,
                        size: 10000
                    }
                ],
                epilogues_unwind_codes: vec![vec![
                    UnwindCode::StackDealloc {
                        offset: 34,
                        size: 10000
                    },
                    UnwindCode::PopRegister {
                        offset: 36,
                        reg: RU::rbp.into()
                    }
                ]],
                code_len: 37,
            }
        );
    }

    #[test]
    #[cfg_attr(feature = "x64", should_panic)] // TODO #2079
    fn test_large_alloc() {
        let isa = lookup(triple!("x86_64"))
            .expect("expect x86 ISA")
            .finish(Flags::new(builder()));

        let mut context = Context::for_function(create_function(
            CallConv::WindowsFastcall,
            Some(StackSlotData::new(StackSlotKind::ExplicitSlot, 1000000)),
        ));

        context.compile(&*isa).expect("expected compilation");

        let unwind = create_unwind_info(&context.func, &*isa, None)
            .expect("can create unwind info")
            .expect("expected unwind info");

        assert_eq!(
            unwind,
            UnwindInfo {
                prologue_size: 27,
                prologue_unwind_codes: vec![
                    UnwindCode::PushRegister {
                        offset: 2,
                        reg: RU::rbp.into(),
                    },
                    UnwindCode::StackAlloc {
                        offset: 27,
                        size: 1000000
                    }
                ],
                epilogues_unwind_codes: vec![vec![
                    UnwindCode::StackDealloc {
                        offset: 34,
                        size: 1000000
                    },
                    UnwindCode::PopRegister {
                        offset: 36,
                        reg: RU::rbp.into()
                    }
                ]],
                code_len: 37,
            }
        );
    }

    fn create_function(call_conv: CallConv, stack_slot: Option<StackSlotData>) -> Function {
        let mut func =
            Function::with_name_signature(ExternalName::user(0, 0), Signature::new(call_conv));

        let block0 = func.dfg.make_block();
        let mut pos = FuncCursor::new(&mut func);
        pos.insert_block(block0);
        pos.ins().return_(&[]);

        if let Some(stack_slot) = stack_slot {
            func.stack_slots.push(stack_slot);
        }

        func
    }

    #[test]
    #[cfg_attr(feature = "x64", should_panic)] // TODO #2079
    fn test_multi_return_func() {
        let isa = lookup(triple!("x86_64"))
            .expect("expect x86 ISA")
            .finish(Flags::new(builder()));

        let mut context = Context::for_function(create_multi_return_function(CallConv::SystemV));

        context.compile(&*isa).expect("expected compilation");

        let unwind = create_unwind_info(&context.func, &*isa, Some(RU::rbp.into()))
            .expect("can create unwind info")
            .expect("expected unwind info");

        assert_eq!(
            unwind,
            UnwindInfo {
                prologue_size: 5,
                prologue_unwind_codes: vec![
                    UnwindCode::PushRegister {
                        offset: 2,
                        reg: RU::rbp.into()
                    },
                    UnwindCode::SetCfaRegister {
                        offset: 5,
                        reg: RU::rbp.into()
                    }
                ],
                epilogues_unwind_codes: vec![
                    vec![
                        UnwindCode::RememberState { offset: 12 },
                        UnwindCode::PopRegister {
                            offset: 12,
                            reg: RU::rbp.into()
                        },
                        UnwindCode::RestoreState { offset: 13 }
                    ],
                    vec![UnwindCode::PopRegister {
                        offset: 15,
                        reg: RU::rbp.into()
                    }]
                ],
                code_len: 16,
            }
        );
    }

    fn create_multi_return_function(call_conv: CallConv) -> Function {
        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(types::I32));
        let mut func = Function::with_name_signature(ExternalName::user(0, 0), sig);

        let block0 = func.dfg.make_block();
        let v0 = func.dfg.append_block_param(block0, types::I32);
        let block1 = func.dfg.make_block();
        let block2 = func.dfg.make_block();

        let mut pos = FuncCursor::new(&mut func);
        pos.insert_block(block0);
        pos.ins().brnz(v0, block2, &[]);
        pos.ins().jump(block1, &[]);

        pos.insert_block(block1);
        pos.ins().return_(&[]);

        pos.insert_block(block2);
        pos.ins().return_(&[]);

        func
    }
}
