;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

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
;;   ldr x7, [x0, #88]
;;   subs xzr, x2, x7
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x9, [x0, #80]
;;   add x9, x9, #4096
;;   strb w3, [x9, x2]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
;;
;; function u0:1:
;; block0:
;;   ldr x7, [x0, #88]
;;   subs xzr, x2, x7
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x9, [x0, #80]
;;   add x8, x9, #4096
;;   ldrb w0, [x8, x2]
;;   b label2
;; block2:
;;   ret
;; block3:
;;   udf #0xc11f
