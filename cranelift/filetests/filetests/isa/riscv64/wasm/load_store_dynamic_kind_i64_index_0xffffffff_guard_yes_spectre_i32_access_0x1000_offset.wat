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
    i32.store offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; function u0:0:
;; block0:
;;   ld a4,8(a2)
;;   ugt a4,a0,a4##ty=i64
;;   ld a5,0(a2)
;;   add a5,a5,a0
;;   lui a0,1
;;   add a5,a5,a0
;;   li a0,0
;;   sltu a2,zero,a4
;;   sub a2,zero,a2
;;   and a3,a0,a2
;;   not a0,a2
;;   and a2,a5,a0
;;   or a3,a3,a2
;;   sw a1,0(a3)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   ld a4,8(a1)
;;   ugt a4,a0,a4##ty=i64
;;   ld a5,0(a1)
;;   add a5,a5,a0
;;   lui a0,1
;;   add a5,a5,a0
;;   li a0,0
;;   sltu a1,zero,a4
;;   sub a1,zero,a1
;;   and a3,a0,a1
;;   not a0,a1
;;   and a1,a5,a0
;;   or a3,a3,a1
;;   lw a0,0(a3)
;;   j label1
;; block1:
;;   ret
