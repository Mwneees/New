;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store offset=0)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0))

;; wasm[0]::function[0]:
;;    0: stp     x29, x30, [sp, #-0x10]!
;;    4: mov     x29, sp
;;    8: ldr     x10, [x0, #0x58]
;;    c: ldr     x11, [x0, #0x50]
;;   10: mov     w12, w2
;;   14: mov     x13, #0
;;   18: add     x11, x11, w2, uxtw
;;   1c: cmp     x12, x10
;;   20: csel    x11, x13, x11, hi
;;   24: csdb
;;   28: str     w3, [x11]
;;   2c: ldp     x29, x30, [sp], #0x10
;;   30: ret
;;
;; wasm[0]::function[1]:
;;   40: stp     x29, x30, [sp, #-0x10]!
;;   44: mov     x29, sp
;;   48: ldr     x10, [x0, #0x58]
;;   4c: ldr     x11, [x0, #0x50]
;;   50: mov     w12, w2
;;   54: mov     x13, #0
;;   58: add     x11, x11, w2, uxtw
;;   5c: cmp     x12, x10
;;   60: csel    x11, x13, x11, hi
;;   64: csdb
;;   68: ldr     w0, [x11]
;;   6c: ldp     x29, x30, [sp], #0x10
;;   70: ret
