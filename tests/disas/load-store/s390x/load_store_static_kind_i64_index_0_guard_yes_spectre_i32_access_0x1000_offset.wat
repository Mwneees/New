;;! target = "s390x"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; function u0:0:
;;   lg %r1, 8(%r2)
;;   lg %r1, 0(%r1)
;;   clgrtle %r15, %r1
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 160, offset_downward_to_clobbers: 0 }
;;   stmg %r14, %r15, 112(%r15)
;;   unwind SaveReg { clobber_offset: 112, reg: p14i }
;;   unwind SaveReg { clobber_offset: 120, reg: p15i }
;;   unwind StackAlloc { size: 0 }
;; block0:
;;   lghi %r14, 0
;;   lgr %r3, %r4
;;   ag %r3, 80(%r2)
;;   aghi %r3, 4096
;;   clgfi %r4, 4294963196
;;   locgrh %r3, %r14
;;   strv %r5, 0(%r3)
;;   jg label1
;; block1:
;;   lmg %r14, %r15, 112(%r15)
;;   br %r14
;;
;; function u0:1:
;;   lg %r1, 8(%r2)
;;   lg %r1, 0(%r1)
;;   clgrtle %r15, %r1
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 160, offset_downward_to_clobbers: 0 }
;;   unwind StackAlloc { size: 0 }
;; block0:
;;   lghi %r3, 0
;;   lgr %r5, %r4
;;   ag %r5, 80(%r2)
;;   aghi %r5, 4096
;;   clgfi %r4, 4294963196
;;   locgrh %r5, %r3
;;   lrv %r2, 0(%r5)
;;   jg label1
;; block1:
;;   br %r14
