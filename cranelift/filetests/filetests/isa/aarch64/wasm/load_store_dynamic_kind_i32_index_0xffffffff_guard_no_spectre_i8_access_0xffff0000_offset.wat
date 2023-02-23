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
;;! offset_guard_size = 0xffffffff
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
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; function u0:0:
;; block0:
;;   mov w10, w0
;;   movn w9, #65534
;;   adds x11, x10, x9
;;   b.lo 8 ; udf
;;   ldr x12, [x2, #8]
;;   subs xzr, x11, x12
;;   b.hi label1 ; b label2
;; block2:
;;   ldr x14, [x2]
;;   movz x15, #65535, LSL #16
;;   add x14, x15, x14
;;   strb w1, [x14, w0, UXTW]
;;   b label3
;; block3:
;;   ret
;; block1:
;;   udf #0xc11f
;;
;; function u0:1:
;; block0:
;;   mov w10, w0
;;   movn w9, #65534
;;   adds x11, x10, x9
;;   b.lo 8 ; udf
;;   ldr x12, [x1, #8]
;;   subs xzr, x11, x12
;;   b.hi label1 ; b label2
;; block2:
;;   ldr x14, [x1]
;;   movz x13, #65535, LSL #16
;;   add x13, x13, x14
;;   ldrb w0, [x13, w0, UXTW]
;;   b label3
;; block3:
;;   ret
;; block1:
;;   udf #0xc11f
