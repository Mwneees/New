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
    i32.store offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; function u0:0:
;; block0:
;;   mov w9, w0
;;   ldr x10, [x2, #8]
;;   movz x11, #4100
;;   sub x10, x10, x11
;;   subs xzr, x9, x10
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x11, [x2]
;;   add x11, x11, #4096
;;   str w1, [x11, w0, UXTW]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
;;
;; function u0:1:
;; block0:
;;   mov w9, w0
;;   ldr x10, [x1, #8]
;;   movz x11, #4100
;;   sub x10, x10, x11
;;   subs xzr, x9, x10
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x11, [x1]
;;   add x10, x11, #4096
;;   ldr w0, [x10, w0, UXTW]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
