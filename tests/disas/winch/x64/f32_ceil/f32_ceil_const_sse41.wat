;;! target = "x86_64"
;;! test = "winch"
;;! flags = ["-Ccranelift-has_sse41"]

(module
    (func (result f32)
        (f32.const -1.32)
        (f32.ceil)
    )
)
;; wasm[0]::function[0]:
;;    0: pushq   %rbp
;;    1: movq    %rsp, %rbp
;;    4: movq    8(%rdi), %r11
;;    8: movq    (%r11), %r11
;;    b: addq    $0x10, %r11
;;   12: cmpq    %rsp, %r11
;;   15: ja      0x3f
;;   1b: movq    %rdi, %r14
;;   1e: subq    $0x10, %rsp
;;   22: movq    %rdi, 8(%rsp)
;;   27: movq    %rsi, (%rsp)
;;   2b: movss   0x15(%rip), %xmm0
;;   33: roundss $2, %xmm0, %xmm0
;;   39: addq    $0x10, %rsp
;;   3d: popq    %rbp
;;   3e: retq
;;   3f: ud2
;;   41: addb    %al, (%rax)
;;   43: addb    %al, (%rax)
;;   45: addb    %al, (%rax)
;;   47: addb    %al, %bl
;;   49: cmc
;;   4a: testb   $0xbf, %al
