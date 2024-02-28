;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0xffff0000))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w12, w2
;;       mov     x13, #0
;;       ldr     x14, [x0, #0x60]
;;       add     x14, x14, w2, uxtw
;;       mov     x15, #0xffff0000
;;       add     x14, x14, x15
;;       mov     x11, #0xfffc
;;       cmp     x12, x11
;;       csel    x14, x13, x14, hi
;;       csdb
;;       str     w3, [x14]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w12, w2
;;       mov     x13, #0
;;       ldr     x14, [x0, #0x60]
;;       add     x14, x14, w2, uxtw
;;       mov     x15, #0xffff0000
;;       add     x14, x14, x15
;;       mov     x11, #0xfffc
;;       cmp     x12, x11
;;       csel    x14, x13, x14, hi
;;       csdb
;;       ldr     w0, [x14]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
