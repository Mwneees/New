;;! target = "x86_64"
;;! test = "winch"

(module
    (func (result i32)
        (i32.const 2)
        (i32.const 3)
        (i32.ge_u)
    )
)
;;      	 55                   	pushq	%rbp
;;      	 4889e5               	movq	%rsp, %rbp
;;      	 4c8b5f08             	movq	8(%rdi), %r11
;;      	 4d8b1b               	movq	(%r11), %r11
;;      	 4981c310000000       	addq	$0x10, %r11
;;      	 4939e3               	cmpq	%rsp, %r11
;;      	 0f8727000000         	ja	0x42
;;   1b:	 4989fe               	movq	%rdi, %r14
;;      	 4883ec10             	subq	$0x10, %rsp
;;      	 48897c2408           	movq	%rdi, 8(%rsp)
;;      	 48893424             	movq	%rsi, (%rsp)
;;      	 b802000000           	movl	$2, %eax
;;      	 83f803               	cmpl	$3, %eax
;;      	 b800000000           	movl	$0, %eax
;;      	 400f93c0             	setae	%al
;;      	 4883c410             	addq	$0x10, %rsp
;;      	 5d                   	popq	%rbp
;;      	 c3                   	retq	
;;   42:	 0f0b                 	ud2	
