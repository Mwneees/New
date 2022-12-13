;;! target = "x86_64"
;;!
;;! settings = ['enable_heap_access_spectre_mitigation=false']
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

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-heap-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store offset=0)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0))

;; function u0:0(i32, i32, i64 vmctx) fast {
;;     gv0 = vmctx
;;     gv1 = load.i64 notrap aligned readonly gv0+8
;;     gv2 = load.i64 notrap aligned readonly gv0
;;
;;                                 block0(v0: i32, v1: i32, v2: i64):
;; @0040                               v3 = uextend.i64 v0
;; @0040                               v4 = global_value.i64 gv1
;; @0040                               v5 = iadd_imm v4, -4
;; @0040                               v6 = icmp ugt v3, v5
;; @0040                               trapnz v6, heap_oob
;; @0040                               v7 = global_value.i64 gv2
;; @0040                               v8 = iadd v7, v3
;; @0040                               store little heap v1, v8
;; @0043                               jump block1
;;
;;                                 block1:
;; @0043                               return
;; }
;;
;; function u0:1(i32, i64 vmctx) -> i32 fast {
;;     gv0 = vmctx
;;     gv1 = load.i64 notrap aligned readonly gv0+8
;;     gv2 = load.i64 notrap aligned readonly gv0
;;
;;                                 block0(v0: i32, v1: i64):
;; @0048                               v3 = uextend.i64 v0
;; @0048                               v4 = global_value.i64 gv1
;; @0048                               v5 = iadd_imm v4, -4
;; @0048                               v6 = icmp ugt v3, v5
;; @0048                               trapnz v6, heap_oob
;; @0048                               v7 = global_value.i64 gv2
;; @0048                               v8 = iadd v7, v3
;; @0048                               v9 = load.i32 little heap v8
;; @004b                               jump block1(v9)
;;
;;                                 block1(v2: i32):
;; @004b                               return v2
;; }