;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-forced -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

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
;;    0: pushq   %rbp
;;    1: movq    %rsp, %rbp
;;    4: xorq    %rsi, %rsi
;;    7: movq    %rdx, %rax
;;    a: addq    0x50(%rdi), %rax
;;    e: movl    $0xffff0000, %edi
;;   13: leaq    (%rax, %rdi), %r11
;;   17: cmpq    $0xffff, %rdx
;;   1e: cmovaq  %rsi, %r11
;;   22: movb    %cl, (%r11)
;;   25: movq    %rbp, %rsp
;;   28: popq    %rbp
;;   29: retq
;;
;; wasm[0]::function[1]:
;;   30: pushq   %rbp
;;   31: movq    %rsp, %rbp
;;   34: xorq    %rsi, %rsi
;;   37: movq    %rdx, %rax
;;   3a: addq    0x50(%rdi), %rax
;;   3e: movl    $0xffff0000, %edi
;;   43: leaq    (%rax, %rdi), %r11
;;   47: cmpq    $0xffff, %rdx
;;   4e: cmovaq  %rsi, %r11
;;   52: movzbq  (%r11), %rax
;;   56: movq    %rbp, %rsp
;;   59: popq    %rbp
;;   5a: retq
