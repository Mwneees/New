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
    i32.store offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; function u0:0:
;; block0:
;;   uext.w t3,a0
;;   lui a7,65535
;;   addi a7,a7,4092
;;   ugt t0,t3,a7##ty=i64
;;   bne t0,zero,taken(label1),not_taken(label2)
;; block2:
;;   ld t0,0(a2)
;;   add t0,t0,t3
;;   lui t4,1
;;   add t1,t0,t4
;;   sw a1,0(t1)
;;   j label3
;; block3:
;;   ret
;; block1:
;;   udf##trap_code=heap_oob
;;
;; function u0:1:
;; block0:
;;   uext.w t3,a0
;;   lui a7,65535
;;   addi a7,a7,4092
;;   ugt t0,t3,a7##ty=i64
;;   bne t0,zero,taken(label1),not_taken(label2)
;; block2:
;;   ld t0,0(a1)
;;   add t0,t0,t3
;;   lui t4,1
;;   add t1,t0,t4
;;   lw a0,0(t1)
;;   j label3
;; block3:
;;   ret
;; block1:
;;   udf##trap_code=heap_oob
