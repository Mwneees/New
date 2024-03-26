;;! target = "x86_64"
;;! test = "winch"
;;! flags = ["-Ccranelift-has_sse41"]

(module
    (func (param f32) (result f32)
        (local.get 0)
        (f32.trunc)
    )
)
;; wasm[0]::function[0]:
;;    0: pushq   %rbp
;;    1: movq    %rsp, %rbp
;;    4: movq    8(%rdi), %r11
;;    8: movq    (%r11), %r11
;;    b: addq    $0x18, %r11
;;   12: cmpq    %rsp, %r11
;;   15: ja      0x44
;;   1b: movq    %rdi, %r14
;;   1e: subq    $0x18, %rsp
;;   22: movq    %rdi, 0x10(%rsp)
;;   27: movq    %rsi, 8(%rsp)
;;   2c: movss   %xmm0, 4(%rsp)
;;   32: movss   4(%rsp), %xmm0
;;   38: roundss $3, %xmm0, %xmm0
;;   3e: addq    $0x18, %rsp
;;   42: popq    %rbp
;;   43: retq
;;   44: ud2
