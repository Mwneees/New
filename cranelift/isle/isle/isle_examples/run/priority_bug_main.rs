mod priority_bug;

struct Context;
impl priority_bug::Context for Context {
    fn is_zero(&mut self, val: u32) -> Option<u32> {
        if val == 0 {
            Some(val)
        } else {
            None
        }
    }

    fn identity(&mut self, val: u32) -> u32 {
        val
    }
}

fn main() {
    let mut ctx = Context;

    assert_eq!(priority_bug::constructor_test(&mut ctx, 0), Some(2));
    assert_eq!(priority_bug::constructor_test(&mut ctx, 1), Some(1));
}
