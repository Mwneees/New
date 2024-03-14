;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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
;;   pushq   %rbp
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   movq    %rsp, %rbp
;;   movq    8(%rdi), %r10
;;   movq    0(%r10), %r10
;;   cmpq    %rsp, %r10
;;   jnbe #trap=stk_ovf
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   movq    %rdx, %rax
;;   addq    %rax, const(0), %rax
;;   jb #trap=heap_oob
;;   movq    88(%rdi), %r9
;;   xorq    %r8, %r8, %r8
;;   addq    %rdx, 80(%rdi), %rdx
;;   movl    $-65536, %r10d
;;   lea     0(%rdx,%r10,1), %rdx
;;   cmpq    %r9, %rax
;;   cmovnbeq %r8, %rdx, %rdx
;;   movb    %cl, 0(%rdx)
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
;;   movq    8(%rdi), %r10
;;   movq    0(%r10), %r10
;;   cmpq    %rsp, %r10
;;   jnbe #trap=stk_ovf
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   movq    %rdx, %rax
;;   addq    %rax, const(0), %rax
;;   jb #trap=heap_oob
;;   movq    88(%rdi), %r8
;;   xorq    %rcx, %rcx, %rcx
;;   addq    %rdx, 80(%rdi), %rdx
;;   movl    $-65536, %r9d
;;   lea     0(%rdx,%r9,1), %rdx
;;   cmpq    %r8, %rax
;;   cmovnbeq %rcx, %rdx, %rdx
;;   movzbq  0(%rdx), %rax
;;   jmp     label1
;; block1:
;;   movq    %rbp, %rsp
;;   popq    %rbp
;;   ret
