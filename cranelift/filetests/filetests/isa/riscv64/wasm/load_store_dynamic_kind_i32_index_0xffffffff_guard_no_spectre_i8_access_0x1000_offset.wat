;;! target = "riscv64"
;;!
;;! settings = ['enable_heap_access_spectre_mitigation=false']
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
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; function u0:0:
;; block0:
;;   slli t0,a0,32
;;   srli t2,t0,32
;;   ld t1,8(a2)
;;   lui t0,1048575
;;   addi t0,t0,4095
;;   add a0,t1,t0
;;   ugt t1,t2,a0##ty=i64
;;   bne t1,zero,taken(label1),not_taken(label2)
;; block2:
;;   ld a0,0(a2)
;;   add a0,a0,t2
;;   lui t2,1
;;   add a2,a0,t2
;;   sb a1,0(a2)
;;   j label3
;; block3:
;;   ret
;; block1:
;;   udf##trap_code=heap_oob
;;
;; function u0:1:
;; block0:
;;   slli t0,a0,32
;;   srli t2,t0,32
;;   ld t1,8(a1)
;;   lui t0,1048575
;;   addi t0,t0,4095
;;   add a0,t1,t0
;;   ugt t1,t2,a0##ty=i64
;;   bne t1,zero,taken(label1),not_taken(label2)
;; block2:
;;   ld a0,0(a1)
;;   add a0,a0,t2
;;   lui t2,1
;;   add a1,a0,t2
;;   lbu a0,0(a1)
;;   j label3
;; block3:
;;   ret
;; block1:
;;   udf##trap_code=heap_oob
