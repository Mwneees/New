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
;;   li t4,0xffff0001
;;   add t1,a0,t4
;;   ult a3,t1,a0##ty=i64
;;   trap_if a3,heap_oob
;;   ld t2,8(a2)
;;   ld a2,0(a2)
;;   add a0,a2,a0
;;   li a2,0xffff0000
;;   add a0,a0,a2
;;   ugt t1,t1,t2##ty=i64
;;   li a2,0
;;   selectif_spectre_guard t2,a2,a0##test=t1
;;   sb a1,0(t2)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   li t4,0xffff0001
;;   add t1,a0,t4
;;   ult a2,t1,a0##ty=i64
;;   trap_if a2,heap_oob
;;   ld t2,8(a1)
;;   ld a1,0(a1)
;;   add a0,a1,a0
;;   li a1,0xffff0000
;;   add a0,a0,a1
;;   ugt t1,t1,t2##ty=i64
;;   li a1,0
;;   selectif_spectre_guard t2,a1,a0##test=t1
;;   lbu a0,0(t2)
;;   j label1
;; block1:
;;   ret
