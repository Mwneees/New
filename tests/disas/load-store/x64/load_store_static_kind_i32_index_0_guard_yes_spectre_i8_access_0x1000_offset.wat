;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    %rdi, %rax
;;       movl    %edx, %edi
;;       xorq    %rsi, %rsi
;;       movq    %rax, %rdx
;;       movq    0x60(%rdx), %rax
;;       leaq    0x1000(%rax, %rdi), %r11
;;       cmpq    0xe(%rip), %rdi
;;       cmovaq  %rsi, %r11
;;       movb    %cl, (%r11)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   2e: addb    %al, (%rax)
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    %rdi, %rcx
;;       movl    %edx, %edi
;;       xorq    %rsi, %rsi
;;       movq    0x60(%rcx), %rax
;;       leaq    0x1000(%rax, %rdi), %r11
;;       cmpq    0x11(%rip), %rdi
;;       cmovaq  %rsi, %r11
;;       movzbq  (%r11), %rax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   6c: addb    %al, (%rax)
;;   6e: addb    %al, (%rax)
