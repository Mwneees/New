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
    i32.store offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0xffff0000))

;; function u0:0:
;; block0:
;;   slli a3,a0,32
;;   srli a4,a3,32
;;   lui a3,262140
;;   addi a3,a3,1
;;   slli a5,a3,2
;;   add a3,a4,a5
;;   trap_if heap_oob##(a3 ult a4)
;;   ld a5,8(a2)
;;   bgtu a3,a5,taken(label3),not_taken(label1)
;; block1:
;;   ld a5,0(a2)
;;   add a5,a5,a4
;;   lui a4,65535
;;   slli a0,a4,4
;;   add a5,a5,a0
;;   sw a1,0(a5)
;;   j label2
;; block2:
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
;;
;; function u0:1:
;; block0:
;;   slli a2,a0,32
;;   srli a4,a2,32
;;   lui a2,262140
;;   addi a3,a2,1
;;   slli a5,a3,2
;;   add a3,a4,a5
;;   trap_if heap_oob##(a3 ult a4)
;;   ld a5,8(a1)
;;   bgtu a3,a5,taken(label3),not_taken(label1)
;; block1:
;;   ld a5,0(a1)
;;   add a5,a5,a4
;;   lui a4,65535
;;   slli a0,a4,4
;;   add a5,a5,a0
;;   lw a0,0(a5)
;;   j label2
;; block2:
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
