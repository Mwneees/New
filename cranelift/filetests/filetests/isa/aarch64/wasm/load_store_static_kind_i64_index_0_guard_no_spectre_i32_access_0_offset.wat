;;! target = "aarch64"
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
;;! # (no heap_bound global for static heaps)
;;!
;;! [[heaps]]
;;! base = "heap_base"
;;! min_size = 0x10000
;;! offset_guard_size = 0
;;! index_type = "i64"
;;! style = { kind = "static", bound = 0x10000000 }

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

;; function u0:0:
;; block0:
;;   orr x5, xzr, #268435452
;;   subs xzr, x0, x5
;;   b.hi label1 ; b label2
;; block2:
;;   ldr x8, [x2]
;;   str w1, [x8, x0]
;;   b label3
;; block3:
;;   ret
;; block1:
;;   udf #0xc11f
;;
;; function u0:1:
;; block0:
;;   orr x5, xzr, #268435452
;;   subs xzr, x0, x5
;;   b.hi label1 ; b label2
;; block2:
;;   ldr x8, [x1]
;;   ldr w0, [x8, x0]
;;   b label3
;; block3:
;;   ret
;; block1:
;;   udf #0xc11f
