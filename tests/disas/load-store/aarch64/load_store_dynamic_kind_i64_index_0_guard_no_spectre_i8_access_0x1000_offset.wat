;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x9, [x0, #0x58]
;;       mov     x10, #0x1001
;;       sub     x9, x9, x10
;;       cmp     x2, x9
;;       b.hi    #0x30
;;       ldr     x11, [x0, #0x50]
;;       add     x11, x11, #1, lsl #12
;;       strb    w3, [x11, x2]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;       .byte   0x1f, 0xc1, 0x00, 0x00
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x9, [x0, #0x58]
;;       mov     x10, #0x1001
;;       sub     x9, x9, x10
;;       cmp     x2, x9
;;       b.hi    #0x70
;;       ldr     x11, [x0, #0x50]
;;       add     x10, x11, #1, lsl #12
;;       ldrb    w0, [x10, x2]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;       .byte   0x1f, 0xc1, 0x00, 0x00
