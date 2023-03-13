;;! target = "s390x"
;;!
;;! settings = ['enable_heap_access_spectre_mitigation=true']
;;!
;;! compile = true
;;!
;;! [globals.vmctx]
;;! type = "i64"
;;! vmctx = true
;;!
;;! [globals.heap_base]
;;! type = "i64"
;;! load = { base = "vmctx", offset = 0, readonly = true }
;;!
;;! [globals.heap_bound]
;;! type = "i64"
;;! load = { base = "vmctx", offset = 8, readonly = true }
;;!
;;! [[heaps]]
;;! base = "heap_base"
;;! min_size = 0x10000
;;! offset_guard_size = 0
;;! index_type = "i32"
;;! style = { kind = "dynamic", bound = "heap_bound" }

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

;; function u0:0:
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 160, offset_downward_to_clobbers: 0 }
;;   stmg %r9, %r15, 72(%r15)
;;   unwind SaveReg { clobber_offset: 72, reg: p9i }
;;   unwind SaveReg { clobber_offset: 80, reg: p10i }
;;   unwind SaveReg { clobber_offset: 88, reg: p11i }
;;   unwind SaveReg { clobber_offset: 96, reg: p12i }
;;   unwind SaveReg { clobber_offset: 104, reg: p13i }
;;   unwind SaveReg { clobber_offset: 112, reg: p14i }
;;   unwind SaveReg { clobber_offset: 120, reg: p15i }
;;   unwind StackAlloc { size: 0 }
;; block0:
;;   lgr %r5, %r4
;;   llgfr %r4, %r2
;;   llilf %r9, 4294901764
;;   algfr %r9, %r2
;;   jnle #trap
;;   lgr %r2, %r5
;;   lg %r5, 8(%r2)
;;   ag %r4, 0(%r2)
;;   llilh %r2, 65535
;;   agr %r4, %r2
;;   lghi %r2, 0
;;   clgr %r9, %r5
;;   locgrh %r4, %r2
;;   strv %r3, 0(%r4)
;;   jg label1
;; block1:
;;   lmg %r9, %r15, 72(%r15)
;;   br %r14
;;
;; function u0:1:
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 160, offset_downward_to_clobbers: 0 }
;;   unwind StackAlloc { size: 0 }
;; block0:
;;   lgr %r4, %r3
;;   llgfr %r3, %r2
;;   llilf %r5, 4294901764
;;   algfr %r5, %r2
;;   jnle #trap
;;   lgr %r2, %r4
;;   lg %r4, 8(%r2)
;;   ag %r3, 0(%r2)
;;   llilh %r2, 65535
;;   agr %r3, %r2
;;   lghi %r2, 0
;;   clgr %r5, %r4
;;   locgrh %r3, %r2
;;   lrv %r2, 0(%r3)
;;   jg label1
;; block1:
;;   br %r14
