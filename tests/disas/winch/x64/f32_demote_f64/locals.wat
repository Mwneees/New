;;! target = "x86_64"
;;! test = "winch"

(module
    (func (result f32)
        (local f64)  

        (local.get 0)
        (f32.demote_f64)
    )
)
;; wasm[0]::function[0]:
;;    0: pushq   %rbp
;;    1: movq    %rsp, %rbp
;;    4: movq    8(%rdi), %r11
;;    8: movq    (%r11), %r11
;;    b: addq    $0x18, %r11
;;   12: cmpq    %rsp, %r11
;;   15: ja      0x43
;;   1b: movq    %rdi, %r14
;;   1e: subq    $0x18, %rsp
;;   22: movq    %rdi, 0x10(%rsp)
;;   27: movq    %rsi, 8(%rsp)
;;   2c: movq    $0, (%rsp)
;;   34: movsd   (%rsp), %xmm0
;;   39: cvtsd2ss %xmm0, %xmm0
;;   3d: addq    $0x18, %rsp
;;   41: popq    %rbp
;;   42: retq
;;   43: ud2
