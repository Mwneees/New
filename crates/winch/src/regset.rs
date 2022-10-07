use crate::isa::reg::Reg;

/// A bit set to track regiter availability
pub(crate) struct RegSet {
    /// Bitset to track general purpose register availability
    gpr: u32,
    /// Bitset to track floating-point register availability
    fpr: u32,
}

impl RegSet {
    /// Create a new register allocator
    pub fn new(gpr: u32, fpr: u32) -> Self {
        Self { gpr, fpr }
    }

    /// Request a general purpose register
    pub fn any_gpr(&mut self) -> Option<Reg> {
        self.gpr_available().then(|| {
            let index = self.gpr.trailing_zeros();
            self.allocate(index);
            Reg::int(index as usize)
        })
    }

    pub fn free_gpr(&mut self, reg: Reg) {
        self.gpr |= reg.hw_enc() as u32;
    }

    fn gpr_available(&self) -> bool {
        self.gpr != 0
    }

    fn allocate(&mut self, index: u32) {
        self.gpr &= !(1 << index);
    }
}
