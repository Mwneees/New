(module
 (import "env" "assert_eq" (func $assert_eq (param i32) (param i32)))
 (func $main
	(local $x i32)
	(local $y i32)
	(local $z i32)
	(local.set $x (i32.const 2))
	(local.set $y (i32.const 3))
	(local.get $x)
	(local.get $y)
	(local.set $z (i32.add))
	(local.get $z)
	(i32.const 5)
	call $assert_eq)
(start $main))
