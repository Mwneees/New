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
;;   slli a6,a0,32
;;   srli t3,a6,32
;;   ld a7,8(a2)
;;   ugt t4,t3,a7##ty=i64
;;   ld a7,0(a2)
;;   add a7,a7,t3
;;   auipc t3,0; ld t3,12(t3); j 12; .8byte 0xffff0000
;;   add a7,a7,t3
;;   li t3,0
;;   andi t0,t4,255
;;   not t2,t0
;;   addi a2,t2,1
;;   or a3,t0,a2
;;   srli a5,a3,63
;;   andi t4,a5,1
;;   addi t4,t4,-1
;;   not t1,t4
;;   and a0,t3,t1
;;   and a2,a7,t4
;;   or a4,a0,a2
;;   sb a1,0(a4)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   slli a6,a0,32
;;   srli t3,a6,32
;;   ld a7,8(a1)
;;   ugt t4,t3,a7##ty=i64
;;   ld a7,0(a1)
;;   add a7,a7,t3
;;   auipc t3,0; ld t3,12(t3); j 12; .8byte 0xffff0000
;;   add a7,a7,t3
;;   li t3,0
;;   andi t0,t4,255
;;   not t2,t0
;;   addi a1,t2,1
;;   or a3,t0,a1
;;   srli a5,a3,63
;;   andi t4,a5,1
;;   addi t4,t4,-1
;;   not t1,t4
;;   and a0,t3,t1
;;   and a2,a7,t4
;;   or a4,a0,a2
;;   lbu a0,0(a4)
;;   j label1
;; block1:
;;   ret
