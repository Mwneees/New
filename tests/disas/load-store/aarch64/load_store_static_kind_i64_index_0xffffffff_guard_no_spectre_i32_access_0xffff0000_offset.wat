;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -W memory64 -O static-memory-forced -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store offset=0xffff0000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0xffff0000))

;; wasm[0]::function[0]:
;;    0: stp     x29, x30, [sp, #-0x10]!
;;    4: mov     x29, sp
;;    8: mov     x8, #0xfffc
;;    c: cmp     x2, x8
;;   10: b.hi    #0x2c
;;   14: ldr     x10, [x0, #0x50]
;;   18: add     x10, x10, x2
;;   1c: mov     x11, #0xffff0000
;;   20: str     w3, [x10, x11]
;;   24: ldp     x29, x30, [sp], #0x10
;;   28: ret
;;   2c: .byte   0x1f, 0xc1, 0x00, 0x00
;;
;; wasm[0]::function[1]:
;;   40: stp     x29, x30, [sp, #-0x10]!
;;   44: mov     x29, sp
;;   48: mov     x8, #0xfffc
;;   4c: cmp     x2, x8
;;   50: b.hi    #0x6c
;;   54: ldr     x10, [x0, #0x50]
;;   58: add     x10, x10, x2
;;   5c: mov     x11, #0xffff0000
;;   60: ldr     w0, [x10, x11]
;;   64: ldp     x29, x30, [sp], #0x10
;;   68: ret
;;   6c: .byte   0x1f, 0xc1, 0x00, 0x00
