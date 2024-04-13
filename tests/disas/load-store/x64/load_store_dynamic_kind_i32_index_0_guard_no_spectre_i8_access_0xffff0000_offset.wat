;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movl    %edx, %r9d
;;       movq    %r9, %rsi
;;       addq    0x2f(%rip), %rsi
;;       jb      0x38
;;   17: movq    0x68(%rdi), %rax
;;       cmpq    %rax, %rsi
;;       ja      0x36
;;   24: addq    0x60(%rdi), %r9
;;       movl    $0xffff0000, %edx
;;       movb    %cl, (%r9, %rdx)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   36: ud2
;;   38: ud2
;;   3a: addb    %al, (%rax)
;;   3c: addb    %al, (%rax)
;;   3e: addb    %al, (%rax)
;;   40: addl    %eax, (%rax)
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movl    %edx, %r9d
;;       movq    %r9, %rsi
;;       addq    0x2f(%rip), %rsi
;;       jb      0x89
;;   67: movq    0x68(%rdi), %rax
;;       cmpq    %rax, %rsi
;;       ja      0x87
;;   74: addq    0x60(%rdi), %r9
;;       movl    $0xffff0000, %ecx
;;       movzbq  (%r9, %rcx), %rax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   87: ud2
;;   89: ud2
;;   8b: addb    %al, (%rax)
;;   8d: addb    %al, (%rax)
;;   8f: addb    %al, (%rcx)
;;   91: addb    %bh, %bh
;;   93: incl    (%rax)
;;   95: addb    %al, (%rax)
