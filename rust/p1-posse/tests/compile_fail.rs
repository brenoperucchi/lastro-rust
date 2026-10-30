//! P1: o compilador recusa mexer no cache enquanto uma operação ainda segura um slot.
#[test]
fn cache_cannot_change_while_an_operation_holds_a_slot() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
