(module
  (func $main (local i32)
    (local.set 0 (i32.const 0))
    (drop (call $inc))
  )
  (func $inc (result i32)
    (i32.const 1)
  )
  (start $main)
)
