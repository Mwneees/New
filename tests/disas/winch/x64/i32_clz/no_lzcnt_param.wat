;;! target = "x86_64"
;;! test = "winch"

(module
    (func (param i32) (result i32)
        (local.get 0)
        (i32.clz)
    )
)
;;      	 55                   	pushq	%rbp
;;      	 4889e5               	movq	%rsp, %rbp
;;      	 4c8b5f08             	movq	8(%rdi), %r11
;;      	 4d8b1b               	movq	(%r11), %r11
;;      	 4981c318000000       	addq	$0x18, %r11
;;      	 4939e3               	cmpq	%rsp, %r11
;;      	 0f8734000000         	ja	0x4f
;;   1b:	 4989fe               	movq	%rdi, %r14
;;      	 4883ec18             	subq	$0x18, %rsp
;;      	 48897c2410           	movq	%rdi, 0x10(%rsp)
;;      	 4889742408           	movq	%rsi, 8(%rsp)
;;      	 89542404             	movl	%edx, 4(%rsp)
;;      	 8b442404             	movl	4(%rsp), %eax
;;      	 0fbdc0               	bsrl	%eax, %eax
;;      	 41bb00000000         	movl	$0, %r11d
;;      	 410f95c3             	setne	%r11b
;;      	 f7d8                 	negl	%eax
;;      	 83c020               	addl	$0x20, %eax
;;      	 4429d8               	subl	%r11d, %eax
;;      	 4883c418             	addq	$0x18, %rsp
;;      	 5d                   	popq	%rbp
;;      	 c3                   	retq	
;;   4f:	 0f0b                 	ud2	
