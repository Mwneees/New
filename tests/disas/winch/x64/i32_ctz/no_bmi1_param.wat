;;! target = "x86_64"
;;! test = "winch"

(module
    (func (param i32) (result i32)
        (local.get 0)
        (i32.ctz)
    )
)
;;      	 55                   	pushq	%rbp
;;      	 4889e5               	movq	%rsp, %rbp
;;      	 4c8b5f08             	movq	8(%rdi), %r11
;;      	 4d8b1b               	movq	(%r11), %r11
;;      	 4981c318000000       	addq	$0x18, %r11
;;      	 4939e3               	cmpq	%rsp, %r11
;;      	 0f8733000000         	ja	0x4e
;;   1b:	 4989fe               	movq	%rdi, %r14
;;      	 4883ec18             	subq	$0x18, %rsp
;;      	 48897c2410           	movq	%rdi, 0x10(%rsp)
;;      	 4889742408           	movq	%rsi, 8(%rsp)
;;      	 89542404             	movl	%edx, 4(%rsp)
;;      	 8b442404             	movl	4(%rsp), %eax
;;      	 0fbcc0               	bsfl	%eax, %eax
;;      	 41bb00000000         	movl	$0, %r11d
;;      	 410f94c3             	sete	%r11b
;;      	 41c1e305             	shll	$5, %r11d
;;      	 4401d8               	addl	%r11d, %eax
;;      	 4883c418             	addq	$0x18, %rsp
;;      	 5d                   	popq	%rbp
;;      	 c3                   	retq	
;;   4e:	 0f0b                 	ud2	
