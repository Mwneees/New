;;! target = "x86_64"
;;!
;;! settings = ['enable_heap_access_spectre_mitigation=true']
;;!
;;! compile = false
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
;;! index_type = "i32"
;;! style = { kind = "static", bound = 0x10000000 }

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

;; function u0:0(i32, i32, i64 vmctx) fast {
;;     gv0 = vmctx
;;     gv1 = load.i64 notrap aligned readonly gv0
;;     heap0 = static gv1, min 0x0001_0000, bound 0x1000_0000, offset_guard 0, index_type i32
;;
;;                                 block0(v0: i32, v1: i32, v2: i64):
;; @0040                               v3 = heap_addr.i64 heap0, v0, 4096, 4
;; @0040                               store little heap v1, v3
;; @0044                               jump block1
;;
;;                                 block1:
;; @0044                               return
;; }
;;
;; function u0:1(i32, i64 vmctx) -> i32 fast {
;;     gv0 = vmctx
;;     gv1 = load.i64 notrap aligned readonly gv0
;;     heap0 = static gv1, min 0x0001_0000, bound 0x1000_0000, offset_guard 0, index_type i32
;;
;;                                 block0(v0: i32, v1: i64):
;; @0049                               v3 = heap_addr.i64 heap0, v0, 4096, 4
;; @0049                               v4 = load.i32 little heap v3
;; @004d                               jump block1(v4)
;;
;;                                 block1(v2: i32):
;; @004d                               return v2
;; }