;;! target = "riscv64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; wasm[0]::function[0]:
;;    0: addi    sp, sp, -0x10
;;    4: sd      ra, 8(sp)
;;    8: sd      s0, 0(sp)
;;    c: mv      s0, sp
;;   10: slli    a2, a2, 0x20
;;   14: srli    a4, a2, 0x20
;;   18: auipc   a5, 0
;;   1c: ld      a5, 0x38(a5)
;;   20: bltu    a5, a4, 0x28
;;   24: ld      a5, 0x50(a0)
;;   28: add     a4, a5, a4
;;   2c: lui     t6, 1
;;   30: add     t6, t6, a4
;;   34: sb      a3, 0(t6)
;;   38: ld      ra, 8(sp)
;;   3c: ld      s0, 0(sp)
;;   40: addi    sp, sp, 0x10
;;   44: ret
;;   48: .byte   0x00, 0x00, 0x00, 0x00
;;   4c: .byte   0x00, 0x00, 0x00, 0x00
;;   50: .byte   0xff, 0xef, 0xff, 0xff
;;   54: .byte   0x00, 0x00, 0x00, 0x00
;;
;; wasm[0]::function[1]:
;;   58: addi    sp, sp, -0x10
;;   5c: sd      ra, 8(sp)
;;   60: sd      s0, 0(sp)
;;   64: mv      s0, sp
;;   68: slli    a2, a2, 0x20
;;   6c: srli    a4, a2, 0x20
;;   70: auipc   a3, 0
;;   74: ld      a3, 0x38(a3)
;;   78: bltu    a3, a4, 0x28
;;   7c: ld      a5, 0x50(a0)
;;   80: add     a4, a5, a4
;;   84: lui     t6, 1
;;   88: add     t6, t6, a4
;;   8c: lbu     a0, 0(t6)
;;   90: ld      ra, 8(sp)
;;   94: ld      s0, 0(sp)
;;   98: addi    sp, sp, 0x10
;;   9c: ret
;;   a0: .byte   0x00, 0x00, 0x00, 0x00
;;   a4: .byte   0x00, 0x00, 0x00, 0x00
;;   a8: .byte   0xff, 0xef, 0xff, 0xff
;;   ac: .byte   0x00, 0x00, 0x00, 0x00
