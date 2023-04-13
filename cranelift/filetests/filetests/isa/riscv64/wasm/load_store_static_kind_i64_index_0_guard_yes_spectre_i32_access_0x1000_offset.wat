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
;;! # (no heap_bound global for static heaps)
;;!
;;! [[heaps]]
;;! base = "heap_base"
;;! min_size = 0x10000
;;! offset_guard_size = 0
;;! index_type = "i64"
;;! style = { kind = "static", bound = 0x10000000 }

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; function u0:0:
;; block0:
;;   lui a3,65535
;;   addi a3,a3,4093
;;   sltu a4,a0,a3
;;   xori a6,a4,1
;;   ld a3,0(a2)
;;   add a3,a3,a0
;;   lui a4,1
;;   add a3,a3,a4
;;   li a4,0
;;   andi t0,a6,255
;;   snez t2,t0
;;   sub a2,zero,t2
;;   and a4,a4,a2
;;   not a5,a2
;;   and a7,a3,a5
;;   or t4,a4,a7
;;   sw a1,0(t4)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   lui a2,65535
;;   addi a2,a2,4093
;;   sltu a4,a0,a2
;;   xori a6,a4,1
;;   ld a3,0(a1)
;;   add a3,a3,a0
;;   lui a4,1
;;   add a3,a3,a4
;;   li a4,0
;;   andi t0,a6,255
;;   snez t2,t0
;;   sub a1,zero,t2
;;   and a4,a4,a1
;;   not a5,a1
;;   and a7,a3,a5
;;   or t4,a4,a7
;;   lw a0,0(t4)
;;   j label1
;; block1:
;;   ret
