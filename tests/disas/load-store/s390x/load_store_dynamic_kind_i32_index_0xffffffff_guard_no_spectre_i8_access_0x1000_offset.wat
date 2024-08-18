;;! target = "s390x"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

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
;;       stmg    %r14, %r15, 0x70(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       lgr     %r6, %r4
;;       lg      %r4, 0x68(%r2)
;;       llgfr   %r3, %r6
;;       clgr    %r3, %r4
;;       jgh     0x42
;;       ag      %r3, 0x60(%r2)
;;       lghi    %r6, 0x1000
;;       stc     %r5, 0(%r6, %r3)
;;       lmg     %r14, %r15, 0x110(%r15)
;;       br      %r14
;;       .byte   0x00, 0x00
;;
;; wasm[0]::function[1]:
;;       stmg    %r14, %r15, 0x70(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       lgr     %r5, %r4
;;       lg      %r4, 0x68(%r2)
;;       llgfr   %r3, %r5
;;       clgr    %r3, %r4
;;       jgh     0x88
;;       ag      %r3, 0x60(%r2)
;;       lghi    %r5, 0x1000
;;       llc     %r2, 0(%r5, %r3)
;;       lmg     %r14, %r15, 0x110(%r15)
;;       br      %r14
;;       .byte   0x00, 0x00
