;;! target = "x86_64"

(module
    (func (result i32)
        (i32.const 1)
        (i32.ctz)
    )
)
;;      	 55                   	pushq	%rbp
;;      	 4889e5               	movq	%rsp, %rbp
;;      	 4c8b5f08             	movq	8(%rdi), %r11
;;      	 4d8b1b               	movq	(%r11), %r11
;;      	 4981c310000000       	addq	$0x10, %r11
;;      	 4939e3               	cmpq	%rsp, %r11
;;      	 0f872f000000         	ja	0x4a
;;   1b:	 4989fe               	movq	%rdi, %r14
;;      	 4883ec10             	subq	$0x10, %rsp
;;      	 48897c2408           	movq	%rdi, 8(%rsp)
;;      	 48893424             	movq	%rsi, (%rsp)
;;      	 b801000000           	movl	$1, %eax
;;      	 0fbcc0               	bsfl	%eax, %eax
;;      	 41bb00000000         	movl	$0, %r11d
;;      	 410f94c3             	sete	%r11b
;;      	 41c1e305             	shll	$5, %r11d
;;      	 4401d8               	addl	%r11d, %eax
;;      	 4883c410             	addq	$0x10, %rsp
;;      	 5d                   	popq	%rbp
;;      	 c3                   	retq	
;;   4a:	 0f0b                 	ud2	
