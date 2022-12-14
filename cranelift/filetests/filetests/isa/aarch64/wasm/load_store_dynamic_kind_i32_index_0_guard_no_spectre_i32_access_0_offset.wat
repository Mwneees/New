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
    i32.store offset=0)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0))

;; function u0:0:
;; block0:
;;   mov w7, w0
;;   ldr x8, [x2, #8]
;;   sub x8, x8, #4
;;   subs xzr, x7, x8
;;   b.ls label1 ; b label3
;; block1:
;;   ldr x9, [x2]
;;   str w1, [x9, w0, UXTW]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
;;
;; function u0:1:
;; block0:
;;   mov w7, w0
;;   ldr x8, [x1, #8]
;;   sub x8, x8, #4
;;   subs xzr, x7, x8
;;   b.ls label1 ; b label3
;; block1:
;;   ldr x9, [x1]
;;   ldr w0, [x9, w0, UXTW]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f