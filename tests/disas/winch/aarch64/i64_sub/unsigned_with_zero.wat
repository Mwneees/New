;;! target = "aarch64"
;;! test = "winch"

(module
    (func (result i64)
        (i64.const 1)
     	(i64.const 0)
    	(i64.sub)
    )
)
;; wasm[0]::function[0]:
;;    0: stp     x29, x30, [sp, #-0x10]!
;;    4: mov     x29, sp
;;    8: mov     x28, sp
;;    c: mov     x9, x0
;;   10: sub     sp, sp, #0x10
;;   14: mov     x28, sp
;;   18: stur    x0, [x28, #8]
;;   1c: stur    x1, [x28]
;;   20: mov     x16, #1
;;   24: mov     x0, x16
;;   28: sub     x0, x0, #0
;;   2c: add     sp, sp, #0x10
;;   30: mov     x28, sp
;;   34: ldp     x29, x30, [sp], #0x10
;;   38: ret
