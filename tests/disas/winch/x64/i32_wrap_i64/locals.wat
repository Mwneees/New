;;! target = "x86_64"
;;! test = "winch"

(module
    (func (result i32)
        (local i64)

        (local.get 0)
        (i32.wrap_i64)
    )
)
;; wasm[0]::function[0]:
;;    0: pushq   %rbp
;;    1: movq    %rsp, %rbp
;;    4: movq    8(%rdi), %r11
;;    8: movq    (%r11), %r11
;;    b: addq    $0x18, %r11
;;   12: cmpq    %rsp, %r11
;;   15: ja      0x40
;;   1b: movq    %rdi, %r14
;;   1e: subq    $0x18, %rsp
;;   22: movq    %rdi, 0x10(%rsp)
;;   27: movq    %rsi, 8(%rsp)
;;   2c: movq    $0, (%rsp)
;;   34: movq    (%rsp), %rax
;;   38: movl    %eax, %eax
;;   3a: addq    $0x18, %rsp
;;   3e: popq    %rbp
;;   3f: retq
;;   40: ud2
