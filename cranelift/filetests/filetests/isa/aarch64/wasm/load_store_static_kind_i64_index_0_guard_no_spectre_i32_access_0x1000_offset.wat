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
    i32.store offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; function u0:0:
;; block0:
;;   movz w7, #61436
;;   movk w7, w7, #4095, LSL #16
;;   subs xzr, x0, x7
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x9, [x2]
;;   add x10, x0, #4096
;;   str w1, [x10, x9]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
;;
;; function u0:1:
;; block0:
;;   movz w7, #61436
;;   movk w7, w7, #4095, LSL #16
;;   subs xzr, x0, x7
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x9, [x1]
;;   add x8, x0, #4096
;;   ldr w0, [x8, x9]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
