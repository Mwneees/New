;;! target = "riscv64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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
;;   addi sp,sp,-16
;;   sd ra,8(sp)
;;   sd fp,0(sp)
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mv fp,sp
;;   ld t6,8(a0)
;;   ld t6,0(t6)
;;   trap_if stk_ovf##(sp ult t6)
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   ld a1,[const(0)]
;;   add a1,a2,a1
;;   trap_if heap_oob##(a1 ult a2)
;;   ld a4,88(a0)
;;   bgtu a1,a4,taken(label3),not_taken(label1)
;; block1:
;;   ld a4,80(a0)
;;   add a2,a4,a2
;;   lui a1,65535
;;   slli a4,a1,4
;;   add a2,a2,a4
;;   sb a3,0(a2)
;;   j label2
;; block2:
;;   ld ra,8(sp)
;;   ld fp,0(sp)
;;   addi sp,sp,16
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
;;
;; function u0:1:
;;   addi sp,sp,-16
;;   sd ra,8(sp)
;;   sd fp,0(sp)
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mv fp,sp
;;   ld t6,8(a0)
;;   ld t6,0(t6)
;;   trap_if stk_ovf##(sp ult t6)
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   ld a1,[const(0)]
;;   add a1,a2,a1
;;   trap_if heap_oob##(a1 ult a2)
;;   ld a3,88(a0)
;;   bgtu a1,a3,taken(label3),not_taken(label1)
;; block1:
;;   ld a3,80(a0)
;;   add a2,a3,a2
;;   lui a1,65535
;;   slli a3,a1,4
;;   add a2,a2,a3
;;   lbu a0,0(a2)
;;   j label2
;; block2:
;;   ld ra,8(sp)
;;   ld fp,0(sp)
;;   addi sp,sp,16
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
