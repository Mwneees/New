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
    i32.store offset=0)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0))

;; function u0:0:
;; block0:
;;   slli a3,a0,32
;;   srli a4,a3,32
;;   ld a3,8(a2)
;;   ugt a5,a4,a3##ty=i64
;;   ld a3,0(a2)
;;   add a3,a3,a4
;;   li a4,0
;;   sub a5,zero,a5
;;   and a2,a4,a5
;;   not a4,a5
;;   and a5,a3,a4
;;   or a2,a2,a5
;;   sw a1,0(a2)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   slli a2,a0,32
;;   srli a4,a2,32
;;   ld a3,8(a1)
;;   ugt a5,a4,a3##ty=i64
;;   ld a3,0(a1)
;;   add a3,a3,a4
;;   li a4,0
;;   sub a5,zero,a5
;;   and a1,a4,a5
;;   not a4,a5
;;   and a5,a3,a4
;;   or a1,a1,a5
;;   lw a0,0(a1)
;;   j label1
;; block1:
;;   ret
