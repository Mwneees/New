;;! target = "x86_64"
;;! test = "winch"

(module
    (func (result f64)
        (i64.const 1)
        (f64.convert_i64_s)
    )
)
;; wasm[0]::function[0]:
;;    0: pushq   %rbp
;;    1: movq    %rsp, %rbp
;;    4: movq    8(%rdi), %r11
;;    8: movq    (%r11), %r11
;;    b: addq    $0x10, %r11
;;   12: cmpq    %rsp, %r11
;;   15: ja      0x3d
;;   1b: movq    %rdi, %r14
;;   1e: subq    $0x10, %rsp
;;   22: movq    %rdi, 8(%rsp)
;;   27: movq    %rsi, (%rsp)
;;   2b: movq    $1, %rax
;;   32: cvtsi2sdq %rax, %xmm0
;;   37: addq    $0x10, %rsp
;;   3b: popq    %rbp
;;   3c: retq
;;   3d: ud2
