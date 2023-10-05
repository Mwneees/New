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
    i32.store8 offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0))

;; function u0:0:
;; block0:
;;   lui a3,65536
;;   addi a4,a3,-1
;;   ugt a4,a0,a4##ty=i64
;;   ld a3,0(a2)
;;   add a3,a3,a0
;;   zero zero
;;   sltu a4,zero,a4
;;   sub a0,zero,a4
;;   and a2,zero,a0
;;   not a4,a0
;;   and a0,a3,a4
;;   or a2,a2,a0
;;   sb a1,0(a2)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   lui a2,65536
;;   addi a4,a2,-1
;;   ugt a4,a0,a4##ty=i64
;;   ld a3,0(a1)
;;   add a3,a3,a0
;;   zero zero
;;   sltu a4,zero,a4
;;   sub a0,zero,a4
;;   and a2,zero,a0
;;   not a4,a0
;;   and a0,a3,a4
;;   or a2,a2,a0
;;   lbu a0,0(a2)
;;   j label1
;; block1:
;;   ret
