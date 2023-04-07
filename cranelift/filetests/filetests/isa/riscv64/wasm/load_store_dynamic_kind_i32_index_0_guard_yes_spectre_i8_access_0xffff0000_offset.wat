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
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; function u0:0:
;; block0:
;;   slli t4,a0,32
;;   srli t1,t4,32
;;   auipc t0,0; ld t0,12(t0); j 12; .8byte 0xffff0001
;;   add t4,t1,t0
;;   ult t2,t4,t1##ty=i64
;;   trap_if t2,heap_oob
;;   ld t2,8(a2)
;;   ugt a0,t4,t2##ty=i64
;;   ld t2,0(a2)
;;   add t1,t2,t1
;;   auipc t2,0; ld t2,12(t2); j 12; .8byte 0xffff0000
;;   add t1,t1,t2
;;   li t2,0
;;   andi a2,a0,255
;;   not a3,a2
;;   addi a5,a3,1
;;   or a7,a2,a5
;;   srli t4,a7,63
;;   andi a0,t4,1
;;   addi a0,a0,-1
;;   not a2,a0
;;   and a4,t2,a2
;;   and a6,t1,a0
;;   or t3,a4,a6
;;   sb a1,0(t3)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   slli t4,a0,32
;;   srli t1,t4,32
;;   auipc t0,0; ld t0,12(t0); j 12; .8byte 0xffff0001
;;   add t4,t1,t0
;;   ult t2,t4,t1##ty=i64
;;   trap_if t2,heap_oob
;;   ld t2,8(a1)
;;   ugt a0,t4,t2##ty=i64
;;   ld t2,0(a1)
;;   add t1,t2,t1
;;   auipc t2,0; ld t2,12(t2); j 12; .8byte 0xffff0000
;;   add t1,t1,t2
;;   li t2,0
;;   andi a1,a0,255
;;   not a3,a1
;;   addi a5,a3,1
;;   or a7,a1,a5
;;   srli t4,a7,63
;;   andi a0,t4,1
;;   addi a0,a0,-1
;;   not a2,a0
;;   and a4,t2,a2
;;   and a6,t1,a0
;;   or t3,a4,a6
;;   lbu a0,0(t3)
;;   j label1
;; block1:
;;   ret
