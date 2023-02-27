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
;;! [globals.heap_bound]
;;! type = "i64"
;;! load = { base = "vmctx", offset = 8, readonly = true }
;;!
;;! [[heaps]]
;;! base = "heap_base"
;;! min_size = 0x10000
;;! offset_guard_size = 0
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
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; function u0:0:
;; block0:
;;   ldr x8, [x2, #8]
;;   movn x7, #4096
;;   add x9, x8, x7
;;   subs xzr, x0, x9
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x10, [x2]
;;   add x11, x0, #4096
;;   strb w1, [x11, x10]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
;;
;; function u0:1:
;; block0:
;;   ldr x8, [x1, #8]
;;   movn x7, #4096
;;   add x9, x8, x7
;;   subs xzr, x0, x9
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x10, [x1]
;;   add x9, x0, #4096
;;   ldrb w0, [x9, x10]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
