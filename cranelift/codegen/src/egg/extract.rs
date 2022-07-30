//! Extraction phase: pick one enode per eclass, avoiding loops.

use super::node::{Node, NodeCtx};
use crate::fx::FxHashMap;
use cranelift_egraph::{EGraph, Id, Language};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EclassState {
    Visiting,
    Visited { cost: u32, node_idx: u32 },
}

#[derive(Clone, Debug)]
pub(crate) struct Extractor {
    eclass_state: FxHashMap<Id, EclassState>,
}

impl Extractor {
    pub(crate) fn new() -> Self {
        Self {
            eclass_state: FxHashMap::default(),
        }
    }

    /// Visit an eclass. Return `None` if deleted, or Some(cost) if
    /// present.
    pub(crate) fn visit_eclass(
        &mut self,
        egraph: &EGraph<NodeCtx>,
        id: Id,
        ctx: &NodeCtx,
    ) -> Option<u32> {
        log::trace!("visiting eclass {:?}", id);
        if let Some(state) = self.eclass_state.get(&id) {
            match state {
                EclassState::Visiting => {
                    log::trace!(" -> found a cycle");
                    // Found a cycle!
                    return None;
                }
                EclassState::Visited { cost, .. } => {
                    log::trace!(" -> already visited, cost {}", *cost);
                    return Some(*cost);
                }
            }
        }
        self.eclass_state.insert(id, EclassState::Visiting);

        let mut best_cost_and_node = None;
        for (node_idx, node) in egraph.enodes(id).iter().enumerate() {
            log::trace!("eclass {:?} has node {:?}", id, node,);
            let this_cost = self.visit_enode(egraph, node, ctx);
            log::trace!("eclass {:?} node {:?} has cost {:?}", id, node, this_cost);
            best_cost_and_node = match (best_cost_and_node, this_cost) {
                (None, None) => None,
                (None, Some(c)) => Some((c, node_idx as u32)),
                (Some((c1, _)), Some(c2)) if c2 < c1 => Some((c2, node_idx as u32)),
                (Some((c1, node_idx1)), _) => Some((c1, node_idx1)),
            };
        }

        match best_cost_and_node {
            Some((cost, node_idx)) => {
                log::trace!("eclass {:?} now visited, with final cost {}", id, cost);
                self.eclass_state
                    .insert(id, EclassState::Visited { cost, node_idx });
                Some(cost)
            }
            None => {
                log::trace!("eclass {:?} has no remaining enodes without cycles", id);
                self.eclass_state.remove(&id);
                None
            }
        }
    }

    fn visit_enode(&mut self, egraph: &EGraph<NodeCtx>, node: &Node, ctx: &NodeCtx) -> Option<u32> {
        let mut cost = node.cost() as u32;
        for &arg in ctx.children(node) {
            log::trace!("node {:?} has child {:?}", node, arg);
            let arg_cost = self.visit_eclass(egraph, arg, ctx)?;
            cost += arg_cost;
        }
        Some(cost)
    }

    pub(crate) fn get_node<'a>(&'a self, egraph: &'a EGraph<NodeCtx>, id: Id) -> Option<&'a Node> {
        match self.eclass_state.get(&id)? {
            &EclassState::Visiting => unreachable!(),
            &EclassState::Visited { node_idx, .. } => Some(&egraph.enodes(id)[node_idx as usize]),
        }
    }
}
