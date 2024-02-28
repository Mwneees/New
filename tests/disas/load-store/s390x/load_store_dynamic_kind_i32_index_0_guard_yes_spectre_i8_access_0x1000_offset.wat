;;! target = "s390x"
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
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; wasm[0]::function[0]:
;;       stmg    %r8, %r15, 0x40(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       lgr     %r3, %r4
;;       lg      %r4, 0x68(%r2)
;;       llgfr   %r3, %r3
;;       aghik   %r8, %r4, -0x1001
;;       lghi    %r4, 0
;;       lgr     %r9, %r3
;;       ag      %r9, 0x60(%r2)
;;       aghik   %r2, %r9, 0x1000
;;       clgr    %r3, %r8
;;       locgrh  %r2, %r4
;;       stc     %r5, 0(%r2)
;;       lmg     %r8, %r15, 0xe0(%r15)
;;       br      %r14
;;
;; wasm[0]::function[1]:
;;       stmg    %r8, %r15, 0x40(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       lg      %r3, 0x68(%r2)
;;       llgfr   %r5, %r4
;;       aghik   %r4, %r3, -0x1001
;;       lghi    %r3, 0
;;       lgr     %r8, %r5
;;       ag      %r8, 0x60(%r2)
;;       aghik   %r2, %r8, 0x1000
;;       clgr    %r5, %r4
;;       locgrh  %r2, %r3
;;       llc     %r2, 0(%r2)
;;       lmg     %r8, %r15, 0xe0(%r15)
;;       br      %r14
