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
;;! offset_guard_size = 0xffffffff
;;! index_type = "i64"
;;! style = { kind = "dynamic", bound = "heap_bound" }

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; function u0:0:
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 160, offset_downward_to_clobbers: 0 }
;;   stmg %r7, %r15, 56(%r15)
;;   unwind SaveReg { clobber_offset: 56, reg: p7i }
;;   unwind SaveReg { clobber_offset: 64, reg: p8i }
;;   unwind SaveReg { clobber_offset: 72, reg: p9i }
;;   unwind SaveReg { clobber_offset: 80, reg: p10i }
;;   unwind SaveReg { clobber_offset: 88, reg: p11i }
;;   unwind SaveReg { clobber_offset: 96, reg: p12i }
;;   unwind SaveReg { clobber_offset: 104, reg: p13i }
;;   unwind SaveReg { clobber_offset: 112, reg: p14i }
;;   unwind SaveReg { clobber_offset: 120, reg: p15i }
;;   unwind StackAlloc { size: 0 }
;; block0:
;;   lg %r8, 8(%r4)
;;   lg %r4, 0(%r4)
;;   agrk %r5, %r4, %r2
;;   llilh %r4, 65535
;;   agr %r5, %r4
;;   lghi %r7, 0
;;   clgr %r2, %r8
;;   locgrh %r5, %r7
;;   stc %r3, 0(%r5)
;;   jg label1
;; block1:
;;   lmg %r7, %r15, 56(%r15)
;;   br %r14
;;
;; function u0:1:
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 160, offset_downward_to_clobbers: 0 }
;;   unwind StackAlloc { size: 0 }
;; block0:
;;   lg %r4, 8(%r3)
;;   lg %r5, 0(%r3)
;;   agr %r5, %r2
;;   llilh %r3, 65535
;;   agrk %r3, %r5, %r3
;;   lghi %r5, 0
;;   clgr %r2, %r4
;;   locgrh %r3, %r5
;;   llc %r2, 0(%r3)
;;   jg label1
;; block1:
;;   br %r14
