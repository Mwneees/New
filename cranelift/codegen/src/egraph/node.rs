//! Node definition for EGraph representation.

use super::MemoryState;
use crate::ir::{Block, Inst, InstructionImms, Opcode, RelSourceLoc, Type};
use crate::loop_analysis::LoopLevel;
use cranelift_egraph::{BumpArena, BumpSlice, CtxEq, CtxHash, Id, Language, UnionFind};
use cranelift_entity::{EntityList, ListPool};
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub enum Node {
    /// A blockparam. Effectively an input/root; does not refer to
    /// predecessors' branch arguments, because this would create
    /// cycles.
    Param {
        /// CLIF block this param comes from.
        block: Block,
        /// Index of blockparam within block.
        index: u32,
        /// Type of the value.
        ty: Type,
    },
    /// A CLIF instruction that is pure (has no side-effects). Not
    /// tied to any location; we will compute a set of locations at
    /// which to compute this node during lowering back out of the
    /// egraph.
    Pure {
        /// The instruction data, without SSA values.
        op: InstructionImms,
        /// eclass arguments to the operator.
        args: EntityList<Id>,
        /// Types of results.
        types: BumpSlice<Type>,
    },
    /// A CLIF instruction that has side-effects or is otherwise not
    /// representable by `Pure`.
    Inst {
        /// The instruction data, without SSA values.
        op: InstructionImms,
        /// eclass arguments to the operator.
        args: EntityList<Id>,
        /// Types of results.
        types: BumpSlice<Type>,
        /// The original instruction. We include this so that the
        /// `Inst`s are not deduplicated: every instance is a
        /// logically separate and unique side-effect. However,
        /// because we clear the DataFlowGraph before elaboration,
        /// this `Inst` is *not* valid to fetch any details from the
        /// original instruction.
        inst: Inst,
        /// The source location to preserve.
        srcloc: RelSourceLoc,
    },
    /// A projection of one result of an `Inst` or `Pure`.
    Result {
        /// `Inst` or `Pure` node.
        value: Id,
        /// Index of the result we want.
        result: usize,
        /// Type of the value.
        ty: Type,
    },

    /// A load instruction. Nominally a side-effecting `Inst` (and
    /// included in the list of side-effecting roots so it will always
    /// be elaborated), but represented as a distinct kind of node so
    /// that we can leverage deduplication to do
    /// redundant-load-elimination for free (and make store-to-load
    /// forwarding much easier).
    Load {
        // -- identity depends on:
        /// The original load operation. Must have one argument, the
        /// address.
        op: InstructionImms,
        /// The type of the load result.
        ty: Type,
        /// Address argument. Actual address has an offset, which is
        /// included in `op` (and thus already considered as part of
        /// the key).
        addr: Id,
        /// The abstract memory state that this load accesses.
        mem_state: MemoryState,

        // -- not included in dedup key:
        /// The `Inst` we will use for a trap location for this
        /// load. Excluded from Eq/Hash so that loads that are
        /// identical except for the specific instance will dedup on
        /// top of each other.
        inst: Inst,
        /// Source location, for traps. Not included in Eq/Hash.
        srcloc: RelSourceLoc,
    },
}

impl Node {
    pub(crate) fn is_non_pure(&self) -> bool {
        match self {
            Node::Inst { .. } | Node::Load { .. } => true,
            _ => false,
        }
    }
}

/// Shared pools for type and id lists in nodes.
pub struct NodeCtx {
    /// Arena for result-type arrays.
    pub types: BumpArena<Type>,
    /// Arena for arg eclass-ID lists.
    pub args: ListPool<Id>,
}

impl std::default::Default for NodeCtx {
    fn default() -> Self {
        Self {
            types: BumpArena::default(),
            args: ListPool::new(),
        }
    }
}

impl NodeCtx {
    pub(crate) fn with_capacity(types: usize, args: usize) -> Self {
        Self {
            types: BumpArena::arena_with_capacity(types),
            args: ListPool::with_capacity(args),
        }
    }
}

impl NodeCtx {
    fn id_eq(&self, a: Id, b: Id, uf: &mut UnionFind) -> bool {
        uf.find_and_update(a) == uf.find_and_update(b)
    }

    fn ids_eq(&self, a: &EntityList<Id>, b: &EntityList<Id>, uf: &mut UnionFind) -> bool {
        let a = a.as_slice(&self.args);
        let b = b.as_slice(&self.args);
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(&a, &b)| self.id_eq(a, b, uf))
    }

    fn hash_id<H: Hasher>(&self, a: Id, hash: &mut H, uf: &mut UnionFind) {
        let id = uf.find_and_update(a);
        id.hash(hash);
    }

    fn hash_ids<H: Hasher>(&self, a: &EntityList<Id>, hash: &mut H, uf: &mut UnionFind) {
        let a = a.as_slice(&self.args);
        for &id in a {
            self.hash_id(id, hash, uf);
        }
    }
}

impl CtxEq<Node, Node> for NodeCtx {
    fn ctx_eq(&self, a: &Node, b: &Node, uf: &mut UnionFind) -> bool {
        match (a, b) {
            (
                &Node::Param { block, index, ty },
                &Node::Param {
                    block: other_block,
                    index: other_index,
                    ty: other_ty,
                },
            ) => block == other_block && index == other_index && ty == other_ty,
            (
                &Node::Result { value, result, ty },
                &Node::Result {
                    value: other_value,
                    result: other_result,
                    ty: other_ty,
                },
            ) => self.id_eq(value, other_value, uf) && result == other_result && ty == other_ty,
            (
                &Node::Pure {
                    ref op,
                    ref args,
                    ref types,
                },
                &Node::Pure {
                    op: ref other_op,
                    args: ref other_args,
                    types: ref other_types,
                },
            ) => {
                *op == *other_op
                    && self.ids_eq(args, other_args, uf)
                    && types.as_slice(&self.types) == other_types.as_slice(&self.types)
            }
            (
                &Node::Inst { inst, ref args, .. },
                &Node::Inst {
                    inst: other_inst,
                    args: ref other_args,
                    ..
                },
            ) => inst == other_inst && self.ids_eq(args, other_args, uf),
            (
                &Node::Load {
                    ref op,
                    ty,
                    addr,
                    mem_state,
                    ..
                },
                &Node::Load {
                    op: ref other_op,
                    ty: other_ty,
                    addr: other_addr,
                    mem_state: other_mem_state,
                    // Explicitly exclude: `inst` and `srcloc`. We
                    // want loads to merge if identical in
                    // opcode/offset, address expression, and last
                    // store (this does implicit
                    // redundant-load-elimination.)
                    ..
                },
            ) => {
                op == other_op
                    && ty == other_ty
                    && self.id_eq(addr, other_addr, uf)
                    && mem_state == other_mem_state
            }
            _ => false,
        }
    }
}

impl CtxHash<Node> for NodeCtx {
    fn ctx_hash(&self, value: &Node, uf: &mut UnionFind) -> u64 {
        let mut state = crate::fx::FxHasher::default();
        std::mem::discriminant(value).hash(&mut state);
        match value {
            &Node::Param {
                block,
                index,
                ty: _,
            } => {
                block.hash(&mut state);
                index.hash(&mut state);
            }
            &Node::Result {
                value,
                result,
                ty: _,
            } => {
                self.hash_id(value, &mut state, uf);
                result.hash(&mut state);
            }
            &Node::Pure {
                ref op,
                ref args,
                types: _,
            } => {
                op.hash(&mut state);
                self.hash_ids(args, &mut state, uf);
                // Don't hash `types`: it requires an indirection
                // (hence cache misses), and result type *should* be
                // fully determined by op and args.
            }
            &Node::Inst { inst, ref args, .. } => {
                inst.hash(&mut state);
                self.hash_ids(args, &mut state, uf);
            }
            &Node::Load {
                ref op,
                ty,
                addr,
                mem_state,
                ..
            } => {
                op.hash(&mut state);
                ty.hash(&mut state);
                self.hash_id(addr, &mut state, uf);
                mem_state.hash(&mut state);
            }
        }

        state.finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Cost(u32);
impl Cost {
    pub(crate) fn at_level(&self, loop_level: LoopLevel) -> Cost {
        let loop_level = std::cmp::min(2, loop_level.level());
        let multiplier = 1u32 << ((10 * loop_level) as u32);
        Cost(self.0.saturating_mul(multiplier))
    }

    pub(crate) fn infinity() -> Cost {
        // 2^32 - 1 is, uh, pretty close to infinite... (we use `Cost`
        // only for heuristics and always saturate so this suffices!)
        Cost(u32::MAX)
    }

    pub(crate) fn zero() -> Cost {
        Cost(0)
    }
}

impl std::default::Default for Cost {
    fn default() -> Cost {
        Cost::zero()
    }
}

impl std::ops::Add<Cost> for Cost {
    type Output = Cost;
    fn add(self, other: Cost) -> Cost {
        // We saturate at one *less* than "infinity".
        let sum = std::cmp::min(u32::MAX - 1, self.0.saturating_add(other.0));
        Cost(sum)
    }
}

pub(crate) fn op_cost(op: &InstructionImms) -> Cost {
    match op.opcode() {
        // Constants.
        Opcode::Iconst | Opcode::F32const | Opcode::F64const | Opcode::Bconst => Cost(0),
        // Extends/reduces.
        Opcode::Bextend
        | Opcode::Breduce
        | Opcode::Uextend
        | Opcode::Sextend
        | Opcode::Ireduce
        | Opcode::Iconcat
        | Opcode::Isplit => Cost(1),
        // "Simple" arithmetic.
        Opcode::Iadd
        | Opcode::Isub
        | Opcode::Band
        | Opcode::BandNot
        | Opcode::Bor
        | Opcode::BorNot
        | Opcode::Bxor
        | Opcode::BxorNot
        | Opcode::Bnot => Cost(2),
        // Everything else.
        _ => Cost(3),
    }
}

impl Language for NodeCtx {
    type Node = Node;

    fn children<'a>(&'a self, node: &'a Node) -> &'a [Id] {
        match node {
            Node::Param { .. } => &[],
            Node::Pure { args, .. } | Node::Inst { args, .. } => args.as_slice(&self.args),
            Node::Load { addr, .. } => std::slice::from_ref(addr),
            Node::Result { value, .. } => std::slice::from_ref(value),
        }
    }

    fn children_mut<'a>(&'a mut self, node: &'a mut Node) -> &'a mut [Id] {
        match node {
            Node::Param { .. } => &mut [],
            Node::Pure { args, .. } | Node::Inst { args, .. } => args.as_mut_slice(&mut self.args),
            Node::Load { addr, .. } => std::slice::from_mut(addr),
            Node::Result { value, .. } => std::slice::from_mut(value),
        }
    }

    fn needs_dedup(&self, node: &Node) -> bool {
        match node {
            Node::Pure { .. } | Node::Load { .. } => true,
            _ => false,
        }
    }
}
