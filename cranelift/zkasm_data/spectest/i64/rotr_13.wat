(module
 (import "env" "assert_eq" (func $assert_eq (param i64) (param i64)))
 (func $main
	i64.const 0x8000000000000000
	i64.const 63
	i64.rotr
	i64.const 1
	call $assert_eq)
 (start $main))
