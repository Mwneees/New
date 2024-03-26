;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w14, w2
;;       mov     w15, #-0xffff
;;       adds    x14, x14, x15
;;       b.hs    #0x48
;;       ldr     x15, [x0, #0x58]
;;       ldr     x1, [x0, #0x50]
;;       mov     x0, #0
;;       add     x1, x1, w2, uxtw
;;       mov     x2, #0xffff0000
;;       add     x1, x1, x2
;;       cmp     x14, x15
;;       csel    x0, x0, x1, hi
;;       csdb
;;       strb    w3, [x0]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;       .byte   0x1f, 0xc1, 0x00, 0x00
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     w14, w2
;;       mov     w15, #-0xffff
;;       adds    x14, x14, x15
;;       b.hs    #0xa8
;;       ldr     x15, [x0, #0x58]
;;       ldr     x1, [x0, #0x50]
;;       mov     x0, #0
;;       add     x1, x1, w2, uxtw
;;       mov     x2, #0xffff0000
;;       add     x1, x1, x2
;;       cmp     x14, x15
;;       csel    x0, x0, x1, hi
;;       csdb
;;       ldrb    w0, [x0]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;       .byte   0x1f, 0xc1, 0x00, 0x00
