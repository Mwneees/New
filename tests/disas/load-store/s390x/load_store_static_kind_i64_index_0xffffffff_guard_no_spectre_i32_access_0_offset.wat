;;! target = "s390x"
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
    i32.store offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0))

;; wasm[0]::function[0]:
;;    0: stmg    %r14, %r15, 0x70(%r15)
;;    6: lgr     %r1, %r15
;;    a: aghi    %r15, -0xa0
;;    e: stg     %r1, 0(%r15)
;;   14: clgfi   %r4, 0xfffffffc
;;   1a: jgh     0x34
;;   20: lg      %r2, 0x50(%r2)
;;   26: strv    %r5, 0(%r4, %r2)
;;   2c: lmg     %r14, %r15, 0x110(%r15)
;;   32: br      %r14
;;   34: .byte   0x00, 0x00
;;
;; wasm[0]::function[1]:
;;   38: stmg    %r14, %r15, 0x70(%r15)
;;   3e: lgr     %r1, %r15
;;   42: aghi    %r15, -0xa0
;;   46: stg     %r1, 0(%r15)
;;   4c: clgfi   %r4, 0xfffffffc
;;   52: jgh     0x6c
;;   58: lg      %r5, 0x50(%r2)
;;   5e: lrv     %r2, 0(%r4, %r5)
;;   64: lmg     %r14, %r15, 0x110(%r15)
;;   6a: br      %r14
;;   6c: .byte   0x00, 0x00
