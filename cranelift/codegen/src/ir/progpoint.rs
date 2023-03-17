//! Program points.

use crate::entity::EntityRef;
use crate::ir::{Block, Inst, ValueDef};
use core::fmt;
use core::u32;

/// A `ProgramPoint` represents a position in a function where the live range of an SSA value can
/// begin or end. It can be either:
///
/// 1. An instruction or
/// 2. A block header.
///
/// This corresponds more or less to the lines in the textual form of Cranelift IR.
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct ProgramPoint(u32);

impl From<Inst> for ProgramPoint {
    fn from(inst: Inst) -> Self {
        let idx = inst.index();
        debug_assert!(idx < (u32::MAX / 2) as usize);
        Self((idx * 2) as u32)
    }
}

impl From<Block> for ProgramPoint {
    fn from(block: Block) -> Self {
        let idx = block.index();
        debug_assert!(idx < (u32::MAX / 2) as usize);
        Self((idx * 2 + 1) as u32)
    }
}

impl From<ValueDef> for ProgramPoint {
    fn from(def: ValueDef) -> Self {
        match def {
            ValueDef::Result(inst, _) => inst.into(),
            ValueDef::Param(block, _) => block.into(),
            ValueDef::Union(_, _) => panic!("Union does not have a single program point"),
        }
    }
}

/// An expanded program point directly exposes the variants, but takes twice the space to
/// represent.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum ExpandedProgramPoint {
    /// An instruction in the function.
    Inst(Inst),
    /// A block header.
    Block(Block),
}

impl ExpandedProgramPoint {
    /// Get the instruction we know is inside.
    pub fn unwrap_inst(self) -> Inst {
        match self {
            Self::Inst(x) => x,
            Self::Block(x) => panic!("expected inst: {}", x),
        }
    }
}

impl From<Inst> for ExpandedProgramPoint {
    fn from(inst: Inst) -> Self {
        Self::Inst(inst)
    }
}

impl From<Block> for ExpandedProgramPoint {
    fn from(block: Block) -> Self {
        Self::Block(block)
    }
}

impl From<ValueDef> for ExpandedProgramPoint {
    fn from(def: ValueDef) -> Self {
        match def {
            ValueDef::Result(inst, _) => inst.into(),
            ValueDef::Param(block, _) => block.into(),
            ValueDef::Union(_, _) => panic!("Union does not have a single program point"),
        }
    }
}

impl From<ProgramPoint> for ExpandedProgramPoint {
    fn from(pp: ProgramPoint) -> Self {
        if pp.0 & 1 == 0 {
            Self::Inst(Inst::from_u32(pp.0 / 2))
        } else {
            Self::Block(Block::from_u32(pp.0 / 2))
        }
    }
}

impl fmt::Display for ExpandedProgramPoint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Inst(x) => write!(f, "{}", x),
            Self::Block(x) => write!(f, "{}", x),
        }
    }
}

impl fmt::Display for ProgramPoint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let epp: ExpandedProgramPoint = (*self).into();
        epp.fmt(f)
    }
}

impl fmt::Debug for ExpandedProgramPoint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ExpandedProgramPoint({})", self)
    }
}

impl fmt::Debug for ProgramPoint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ProgramPoint({})", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityRef;
    use crate::ir::{Block, Inst};
    use alloc::string::ToString;

    #[test]
    fn convert() {
        let i5 = Inst::new(5);
        let b3 = Block::new(3);

        let pp1: ProgramPoint = i5.into();
        let pp2: ProgramPoint = b3.into();

        assert_eq!(pp1.to_string(), "inst5");
        assert_eq!(pp2.to_string(), "block3");
    }
}
