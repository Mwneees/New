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
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; function u0:0:
;; block0:
;;   auipc t4,0; ld t4,12(t4); j 12; .8byte 0xffff0001
;;   add t3,a0,t4
;;   sltu t0,t3,a0
;;   trap_if t0,heap_oob
;;   ld t0,8(a2)
;;   sltu t0,t0,t3
;;   bne t0,zero,taken(label3),not_taken(label1)
;; block1:
;;   ld t1,0(a2)
;;   add t1,t1,a0
;;   auipc t2,0; ld t2,12(t2); j 12; .8byte 0xffff0000
;;   add t1,t1,t2
;;   sb a1,0(t1)
;;   j label2
;; block2:
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
;;
;; function u0:1:
;; block0:
;;   auipc t4,0; ld t4,12(t4); j 12; .8byte 0xffff0001
;;   add t3,a0,t4
;;   sltu t0,t3,a0
;;   trap_if t0,heap_oob
;;   ld t0,8(a1)
;;   sltu t0,t0,t3
;;   bne t0,zero,taken(label3),not_taken(label1)
;; block1:
;;   ld t1,0(a1)
;;   add t1,t1,a0
;;   auipc t2,0; ld t2,12(t2); j 12; .8byte 0xffff0000
;;   add t1,t1,t2
;;   lbu a0,0(t1)
;;   j label2
;; block2:
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
