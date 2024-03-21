;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-forced -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

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
;;   movz x9, #0
;;   ldr x10, [x0, #80]
;;   add x10, x10, x2
;;   orr x8, xzr, #4294967292
;;   subs xzr, x2, x8
;;   csel x11, x9, x10, hi
;;   csdb
;;   str w3, [x11]
;;   b label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   movz x9, #0
;;   ldr x10, [x0, #80]
;;   add x10, x10, x2
;;   orr x8, xzr, #4294967292
;;   subs xzr, x2, x8
;;   csel x11, x9, x10, hi
;;   csdb
;;   ldr w0, [x11]
;;   b label1
;; block1:
;;   ret
