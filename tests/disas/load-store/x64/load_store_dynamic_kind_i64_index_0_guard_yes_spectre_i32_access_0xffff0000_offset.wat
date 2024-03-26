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
    i32.store offset=0xffff0000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0xffff0000))

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    %rdx, %rax
;;       addq    0x2a(%rip), %rax
;;       jb      0x36
;;       movq    0x58(%rdi), %r9
;;       xorq    %r8, %r8
;;       addq    0x50(%rdi), %rdx
;;       movl    $0xffff0000, %r10d
;;       addq    %r10, %rdx
;;       cmpq    %r9, %rax
;;       cmovaq  %r8, %rdx
;;       movl    %ecx, (%rdx)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;       ud2
;;       addb    $0, %al
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    %rdx, %rax
;;       addq    0x2a(%rip), %rax
;;       jb      0x76
;;       movq    0x58(%rdi), %r8
;;       xorq    %rcx, %rcx
;;       addq    0x50(%rdi), %rdx
;;       movl    $0xffff0000, %r9d
;;       addq    %r9, %rdx
;;       cmpq    %r8, %rax
;;       cmovaq  %rcx, %rdx
;;       movl    (%rdx), %eax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;       ud2
;;       addb    $0, %al
