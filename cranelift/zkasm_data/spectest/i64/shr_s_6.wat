(module
 (import "env" "assert_eq" (func $assert_eq (param i64) (param i64)))
 (func $main
	i64.const 0x4000000000000000
	i64.const 1
	i64.shr_s
	i64.const 0x2000000000000000
	call $assert_eq)
 (start $main))
