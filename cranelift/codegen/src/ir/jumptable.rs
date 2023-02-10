//! Jump table representation.
//!
//! Jump tables are declared in the preamble and assigned an `ir::entities::JumpTable` reference.
//! The actual table of destinations is stored in a `JumpTableData` struct defined in this module.

use crate::ir::entities::Block;
use alloc::vec::Vec;
use core::fmt::{self, Display, Formatter};
use core::slice::{Iter, IterMut};

#[cfg(feature = "enable-serde")]
use serde::{Deserialize, Serialize};

/// Contents of a jump table.
///
/// All jump tables use 0-based indexing and are densely populated.
///
/// The default block for the jump table is stored as the last element of the underlying vector,
/// and is not included in the length of the jump table. It can be accessed through the
/// `default_block` function. The default block is iterated via the `iter` and `iter_mut` methods.
#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct JumpTableData {
    // Table entries.
    table: Vec<Block>,
}

impl JumpTableData {
    /// Create a new jump table with the provided blocks
    pub fn new(def: Block, mut table: Vec<Block>) -> Self {
        table.push(def);
        Self { table }
    }

    /// Fetch the default block for this jump table.
    pub fn default_block(&self) -> Block {
        *self.table.last().unwrap()
    }

    /// Mutable access to the default block of this jump table.
    pub fn default_block_mut(&mut self) -> &mut Block {
        self.table.last_mut().unwrap()
    }

    /// The jump table and default block as a single slice. The default block will always be last.
    pub fn all_branches(&self) -> &[Block] {
        self.table.as_slice()
    }

    /// The jump table and default block as a single mutable slice. The default block will always
    /// be last.
    pub fn all_branches_mut(&mut self) -> &mut [Block] {
        self.table.as_mut_slice()
    }

    /// Access the jump table as a slice. This excludes the default block.
    pub fn as_slice(&self) -> &[Block] {
        let last = self.table.len() - 1;
        &self.table.as_slice()[0..last]
    }

    /// Access the jump table as a mutable slice. This excludes the default block.
    pub fn as_mut_slice(&mut self) -> &mut [Block] {
        let last = self.table.len() - 1;
        &mut self.table.as_mut_slice()[0..last]
    }

    /// Returns an iterator to the jump table, excluding the default block.
    #[deprecated(since = "7.0.0", note = "please use `.as_slice()` instead")]
    pub fn iter(&self) -> Iter<Block> {
        self.as_slice().iter()
    }

    /// Returns an iterator that allows modifying each value, excluding the default block.
    #[deprecated(since = "7.0.0", note = "please use `.as_mut_slice()` instead")]
    pub fn iter_mut(&mut self) -> IterMut<Block> {
        self.as_mut_slice().iter_mut()
    }

    /// Clears all entries in this jump table, except for the default block.
    pub fn clear(&mut self) {
        self.table.drain(0..self.table.len() - 1);
    }
}

impl Display for JumpTableData {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        write!(fmt, "{}, [", self.default_block())?;
        if let Some((first, rest)) = self.as_slice().split_first() {
            write!(fmt, "{}", first)?;
            for block in rest {
                write!(fmt, ", {}", block)?;
            }
        }
        write!(fmt, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::JumpTableData;
    use crate::entity::EntityRef;
    use crate::ir::Block;
    use alloc::string::ToString;

    #[test]
    fn empty() {
        let def = Block::new(0);

        let jt = JumpTableData::new(def, vec![]);

        assert_eq!(jt.all_branches().get(0), Some(&def));

        assert_eq!(jt.as_slice().get(0), None);
        assert_eq!(jt.as_slice().get(10), None);

        assert_eq!(jt.to_string(), "block0, []");

        assert_eq!(jt.all_branches(), [def]);
        assert_eq!(jt.as_slice(), []);
    }

    #[test]
    fn insert() {
        let def = Block::new(0);
        let e1 = Block::new(1);
        let e2 = Block::new(2);

        let jt = JumpTableData::new(def, vec![e1, e2, e1]);

        assert_eq!(jt.default_block(), def);
        assert_eq!(jt.to_string(), "block0, [block1, block2, block1]");

        assert_eq!(jt.all_branches(), [e1, e2, e1, def]);
        assert_eq!(jt.as_slice(), [e1, e2, e1]);
    }
}
