;;! target = "riscv64"
;;!
;;! settings = ['enable_heap_access_spectre_mitigation=true']
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
    i32.store offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0))

;; function u0:0:
;; block0:
;;   ld a3,8(a2)
;;   addi a3,a3,-4
;;   sltu a4,a3,a0
;;   ld a2,0(a2)
;;   add a0,a2,a0
;;   li a3,0
;;   andi a7,a4,255
;;   snez t4,a7
;;   sub t1,zero,t4
;;   and a2,a3,t1
;;   not a3,t1
;;   and a4,a0,a3
;;   or a6,a2,a4
;;   sw a1,0(a6)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   ld a2,8(a1)
;;   addi a2,a2,-4
;;   sltu a3,a2,a0
;;   ld a1,0(a1)
;;   add a0,a1,a0
;;   li a2,0
;;   andi a7,a3,255
;;   snez t4,a7
;;   sub t1,zero,t4
;;   and a1,a2,t1
;;   not a2,t1
;;   and a4,a0,a2
;;   or a6,a1,a4
;;   lw a0,0(a6)
;;   j label1
;; block1:
;;   ret
