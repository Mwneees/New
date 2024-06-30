;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x8, [x2, #0x68]
;;       ldr     x10, [x2, #0x60]
;;       mov     x9, #0
;;       add     x10, x10, x4
;;       cmp     x4, x8
;;       csel    x9, x9, x10, hs
;;       csdb
;;       strb    w5, [x9]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x8, [x2, #0x68]
;;       ldr     x10, [x2, #0x60]
;;       mov     x9, #0
;;       add     x10, x10, x4
;;       cmp     x4, x8
;;       csel    x9, x9, x10, hs
;;       csdb
;;       ldrb    w2, [x9]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
