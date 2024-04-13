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
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       cmpq    $0xffff, %rdx
;;       ja      0x24
;;   11: addq    0x60(%rdi), %rdx
;;       movl    $0xffff0000, %r11d
;;       movb    %cl, (%rdx, %r11)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   24: ud2
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       cmpq    $0xffff, %rdx
;;       ja      0x55
;;   41: addq    0x60(%rdi), %rdx
;;       movl    $0xffff0000, %r11d
;;       movzbq  (%rdx, %r11), %rax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   55: ud2
