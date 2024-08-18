;;! target = "s390x"
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
    i32.store offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; wasm[0]::function[0]:
;;       stmg    %r14, %r15, 0x70(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       llgfr   %r6, %r4
;;       lghi    %r4, 0
;;       lgr     %r7, %r6
;;       ag      %r7, 0x60(%r2)
;;       aghik   %r3, %r7, 0x1000
;;       clgfi   %r6, 0xffffeffc
;;       locgrh  %r3, %r4
;;       strv    %r5, 0(%r3)
;;       lmg     %r14, %r15, 0x110(%r15)
;;       br      %r14
;;
;; wasm[0]::function[1]:
;;       stmg    %r14, %r15, 0x70(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       llgfr   %r5, %r4
;;       lghi    %r4, 0
;;       lgr     %r6, %r5
;;       ag      %r6, 0x60(%r2)
;;       aghik   %r3, %r6, 0x1000
;;       clgfi   %r5, 0xffffeffc
;;       locgrh  %r3, %r4
;;       lrv     %r2, 0(%r3)
;;       lmg     %r14, %r15, 0x110(%r15)
;;       br      %r14
