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
;;! # (no heap_bound global for static heaps)
;;!
;;! [[heaps]]
;;! base = "heap_base"
;;! min_size = 0x10000
;;! offset_guard_size = 0xffffffff
;;! index_type = "i32"
;;! style = { kind = "static", bound = 0x10000000 }

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

;; function u0:0:
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 160, offset_downward_to_clobbers: 0 }
;;   unwind StackAlloc { size: 0 }
;; block0:
;;   lgr %r5, %r4
;;   llgfr %r4, %r2
;;   lgr %r2, %r5
;;   ag %r4, 0(%r2)
;;   lghi %r5, 4096
;;   strv %r3, 0(%r5,%r4)
;;   jg label1
;; block1:
;;   br %r14
;;
;; function u0:1:
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 160, offset_downward_to_clobbers: 0 }
;;   unwind StackAlloc { size: 0 }
;; block0:
;;   lgr %r5, %r3
;;   llgfr %r3, %r2
;;   lgr %r2, %r5
;;   ag %r3, 0(%r2)
;;   lghi %r4, 4096
;;   lrv %r2, 0(%r4,%r3)
;;   jg label1
;; block1:
;;   br %r14