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

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    %rdx, %r8
;;       addq    0x32(%rip), %r8
;;       jb      0x37
;;   14: movq    0x68(%rdi), %r9
;;       xorq    %rax, %rax
;;       addq    0x60(%rdi), %rdx
;;       movl    $0xffff0000, %r10d
;;       leaq    (%rdx, %r10), %rdi
;;       cmpq    %r9, %r8
;;       cmovaq  %rax, %rdi
;;       movb    %cl, (%rdi)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   37: ud2
;;   39: addb    %al, (%rax)
;;   3b: addb    %al, (%rax)
;;   3d: addb    %al, (%rax)
;;   3f: addb    %al, (%rcx)
;;   41: addb    %bh, %bh
;;   43: incl    (%rax)
;;   45: addb    %al, (%rax)
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    %rdx, %rcx
;;       addq    0x32(%rip), %rcx
;;       jb      0x99
;;   74: movq    0x68(%rdi), %r8
;;       xorq    %rax, %rax
;;       addq    0x60(%rdi), %rdx
;;       movl    $0xffff0000, %r9d
;;       leaq    (%rdx, %r9), %rdi
;;       cmpq    %r8, %rcx
;;       cmovaq  %rax, %rdi
;;       movzbq  (%rdi), %rax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   99: ud2
;;   9b: addb    %al, (%rax)
;;   9d: addb    %al, (%rax)
;;   9f: addb    %al, (%rcx)
;;   a1: addb    %bh, %bh
;;   a3: incl    (%rax)
;;   a5: addb    %al, (%rax)
