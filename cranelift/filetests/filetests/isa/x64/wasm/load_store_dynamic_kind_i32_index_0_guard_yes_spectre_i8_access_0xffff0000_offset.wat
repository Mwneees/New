;;! target = "x86_64"
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
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; function u0:0:
;;   pushq   %rbp
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   movq    %rsp, %rbp
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   movl    %edi, %r11d
;;   movq    %r11, %rax
;;   addq    %rax, const(0), %rax
;;   jb #trap=heap_oob
;;   movq    8(%rdx), %rcx
;;   addq    %r11, 0(%rdx), %r11
;;   movl    $-65536, %edx
;;   lea     0(%r11,%rdx,1), %rdi
;;   xorq    %rdx, %rdx, %rdx
;;   cmpq    %rcx, %rax
;;   cmovnbeq %rdx, %rdi, %rdi
;;   movb    %sil, 0(%rdi)
;;   jmp     label1
;; block1:
;;   movq    %rbp, %rsp
;;   popq    %rbp
;;   ret
;;
;; function u0:1:
;;   pushq   %rbp
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   movq    %rsp, %rbp
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   movl    %edi, %r11d
;;   movq    %r11, %rax
;;   addq    %rax, const(0), %rax
;;   jb #trap=heap_oob
;;   movq    8(%rsi), %rcx
;;   addq    %r11, 0(%rsi), %r11
;;   movl    $-65536, %edx
;;   lea     0(%r11,%rdx,1), %rdi
;;   xorq    %rdx, %rdx, %rdx
;;   cmpq    %rcx, %rax
;;   cmovnbeq %rdx, %rdi, %rdi
;;   movzbq  0(%rdi), %rax
;;   jmp     label1
;; block1:
;;   movq    %rbp, %rsp
;;   popq    %rbp
;;   ret
