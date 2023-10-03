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
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; function u0:0:
;; block0:
;;   slli a3,a0,32
;;   srli a4,a3,32
;;   ld a3,8(a2)
;;   lui a5,-1
;;   addi a5,a5,-1
;;   add a3,a3,a5
;;   ugt a3,a4,a3##ty=i64
;;   ld a5,0(a2)
;;   add a4,a5,a4
;;   lui a5,1
;;   add a4,a4,a5
;;   li a5,0
;;   sltu a0,zero,a3
;;   sub a0,zero,a0
;;   and a2,a5,a0
;;   not a5,a0
;;   and a0,a4,a5
;;   or a2,a2,a0
;;   sb a1,0(a2)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   slli a2,a0,32
;;   srli a4,a2,32
;;   ld a3,8(a1)
;;   lui a2,-1
;;   addi a5,a2,-1
;;   add a3,a3,a5
;;   ugt a3,a4,a3##ty=i64
;;   ld a5,0(a1)
;;   add a4,a5,a4
;;   lui a5,1
;;   add a4,a4,a5
;;   li a5,0
;;   sltu a0,zero,a3
;;   sub a0,zero,a0
;;   and a2,a5,a0
;;   not a5,a0
;;   and a0,a4,a5
;;   or a2,a2,a0
;;   lbu a0,0(a2)
;;   j label1
;; block1:
;;   ret
