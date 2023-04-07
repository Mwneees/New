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
    i32.store offset=0)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0))

;; function u0:0:
;; block0:
;;   slli a5,a0,32
;;   srli a7,a5,32
;;   lui a6,65536
;;   addi a6,a6,4092
;;   ugt t3,a7,a6##ty=i64
;;   ld a6,0(a2)
;;   add a6,a6,a7
;;   li a7,0
;;   andi t4,t3,255
;;   not t1,t4
;;   addi a0,t1,1
;;   or a2,t4,a0
;;   srli a4,a2,63
;;   andi t3,a4,1
;;   addi t3,t3,-1
;;   not t0,t3
;;   and t2,a7,t0
;;   and a2,a6,t3
;;   or a3,t2,a2
;;   sw a1,0(a3)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   slli a5,a0,32
;;   srli a7,a5,32
;;   lui a6,65536
;;   addi a6,a6,4092
;;   ugt t3,a7,a6##ty=i64
;;   ld a6,0(a1)
;;   add a6,a6,a7
;;   li a7,0
;;   andi t4,t3,255
;;   not t1,t4
;;   addi a0,t1,1
;;   or a2,t4,a0
;;   srli a4,a2,63
;;   andi t3,a4,1
;;   addi t3,t3,-1
;;   not t0,t3
;;   and t2,a7,t0
;;   and a1,a6,t3
;;   or a3,t2,a1
;;   lw a0,0(a3)
;;   j label1
;; block1:
;;   ret
