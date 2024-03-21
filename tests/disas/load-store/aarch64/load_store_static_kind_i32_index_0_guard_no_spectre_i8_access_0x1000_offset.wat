;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; function u0:0:
;; block0:
;;   mov w8, w2
;;   movn w9, #4096
;;   subs xzr, x8, x9
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x10, [x0, #80]
;;   add x10, x10, #4096
;;   strb w3, [x10, w2, UXTW]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
;;
;; function u0:1:
;; block0:
;;   mov w8, w2
;;   movn w9, #4096
;;   subs xzr, x8, x9
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x10, [x0, #80]
;;   add x9, x10, #4096
;;   ldrb w0, [x9, w2, UXTW]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
