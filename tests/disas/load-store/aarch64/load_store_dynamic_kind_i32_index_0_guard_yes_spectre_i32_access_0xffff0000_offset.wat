;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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
;;   stp fp, lr, [sp, #-16]!
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mov fp, sp
;;   ldr x16, [x0, #8]
;;   ldr x16, [x16]
;;   subs xzr, sp, x16, UXTX
;;   b.lo #trap=stk_ovf
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   mov w14, w2
;;   movn w15, #65531
;;   adds x14, x14, x15
;;   b.hs #trap=heap_oob
;;   ldr x15, [x0, #88]
;;   ldr x1, [x0, #80]
;;   movz x0, #0
;;   add x1, x1, x2, UXTW
;;   movz x2, #65535, LSL #16
;;   add x1, x1, x2
;;   subs xzr, x14, x15
;;   csel x0, x0, x1, hi
;;   csdb
;;   str w3, [x0]
;;   b label1
;; block1:
;;   ldp fp, lr, [sp], #16
;;   ret
;;
;; function u0:1:
;;   stp fp, lr, [sp, #-16]!
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mov fp, sp
;;   ldr x16, [x0, #8]
;;   ldr x16, [x16]
;;   subs xzr, sp, x16, UXTX
;;   b.lo #trap=stk_ovf
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   mov w14, w2
;;   movn w15, #65531
;;   adds x14, x14, x15
;;   b.hs #trap=heap_oob
;;   ldr x15, [x0, #88]
;;   ldr x1, [x0, #80]
;;   movz x0, #0
;;   add x1, x1, x2, UXTW
;;   movz x2, #65535, LSL #16
;;   add x1, x1, x2
;;   subs xzr, x14, x15
;;   csel x0, x0, x1, hi
;;   csdb
;;   ldr w0, [x0]
;;   b label1
;; block1:
;;   ldp fp, lr, [sp], #16
;;   ret
