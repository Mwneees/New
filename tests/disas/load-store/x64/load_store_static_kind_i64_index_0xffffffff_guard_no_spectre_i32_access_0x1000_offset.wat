;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -W memory64 -O static-memory-forced -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       cmpq    0x1d(%rip), %rdx
;;       ja      0x22
;;       movq    0x50(%rdi), %r10
;;       movl    %ecx, 0x1000(%r10, %rdx)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;       ud2
;;       addb    %al, (%rax)
;;       addb    %al, (%rax)
;;       cld
;;       outl    %eax, %dx
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       cmpq    0x1d(%rip), %rdx
;;       ja      0x52
;;       movq    0x50(%rdi), %r10
;;       movl    0x1000(%r10, %rdx), %eax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;       ud2
;;       addb    %al, (%rax)
;;       addb    %al, (%rax)
;;       cld
;;       outl    %eax, %dx
