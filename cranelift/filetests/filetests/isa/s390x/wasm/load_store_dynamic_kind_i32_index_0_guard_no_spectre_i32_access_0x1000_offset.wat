;;! target = "s390x"
;;!
;;! settings = ['enable_heap_access_spectre_mitigation=false']
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
;;   lg %r2, 8(%r5)
;;   aghi %r2, -4100
;;   clgr %r4, %r2
;;   jgh label3 ; jg label1
;; block1:
;;   lg %r5, 0(%r5)
;;   agr %r5, %r4
;;   lghi %r2, 4096
;;   strv %r3, 0(%r2,%r5)
;;   jg label2
;; block2:
;;   br %r14
;; block3:
;;   .word 0x0000 # trap=heap_oob
;;
;; function u0:1:
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 160, offset_downward_to_clobbers: 0 }
;;   unwind StackAlloc { size: 0 }
;; block0:
;;   lgr %r4, %r3
;;   llgfr %r3, %r2
;;   lgr %r5, %r4
;;   lg %r4, 8(%r5)
;;   aghi %r4, -4100
;;   clgr %r3, %r4
;;   jgh label3 ; jg label1
;; block1:
;;   lg %r5, 0(%r5)
;;   agr %r5, %r3
;;   lghi %r2, 4096
;;   lrv %r2, 0(%r2,%r5)
;;   jg label2
;; block2:
;;   br %r14
;; block3:
;;   .word 0x0000 # trap=heap_oob
