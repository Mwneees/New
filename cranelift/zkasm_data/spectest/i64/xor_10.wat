(module
 (import "env" "assert_eq" (func $assert_eq (param i64) (param i64)))
 (func $main
	i64.const 0xffffffffffffffff
	i64.const 0xffffffffffffffff
	i64.xor
	i64.const 0
	call $assert_eq)
 (start $main))
