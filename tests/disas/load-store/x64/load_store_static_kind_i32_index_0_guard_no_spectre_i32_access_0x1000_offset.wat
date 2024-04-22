;;! target = "x86_64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0x1000))

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movl    %edx, %r8d
;;       cmpq    0x1a(%rip), %r8
;;       ja      0x25
;;   14: movq    0x60(%rdi), %r10
;;       movl    %ecx, 0x1000(%r10, %r8)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   25: ud2
;;   27: addb    %bh, %ah
;;   29: outl    %eax, %dx
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movl    %edx, %r8d
;;       cmpq    0x1a(%rip), %r8
;;       ja      0x55
;;   44: movq    0x60(%rdi), %r10
;;       movl    0x1000(%r10, %r8), %eax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   55: ud2
;;   57: addb    %bh, %ah
;;   59: outl    %eax, %dx
