;;! target = "x86_64"

(module
    (func (result i64)
        (local $foo i64)

        (i64.const 2)
        (local.set $foo)

        (local.get $foo)
        (i64.ctz)
    )
)
;;      	 55                   	pushq	%rbp
;;      	 4889e5               	movq	%rsp, %rbp
;;      	 4c8b5f08             	movq	8(%rdi), %r11
;;      	 4d8b1b               	movq	(%r11), %r11
;;      	 4981c318000000       	addq	$0x18, %r11
;;      	 4939e3               	cmpq	%rsp, %r11
;;      	 0f8743000000         	ja	0x5e
;;   1b:	 4989fe               	movq	%rdi, %r14
;;      	 4883ec18             	subq	$0x18, %rsp
;;      	 48897c2410           	movq	%rdi, 0x10(%rsp)
;;      	 4889742408           	movq	%rsi, 8(%rsp)
;;      	 48c7042400000000     	movq	$0, (%rsp)
;;      	 48c7c002000000       	movq	$2, %rax
;;      	 48890424             	movq	%rax, (%rsp)
;;      	 488b0424             	movq	(%rsp), %rax
;;      	 480fbcc0             	bsfq	%rax, %rax
;;      	 41bb00000000         	movl	$0, %r11d
;;      	 410f94c3             	sete	%r11b
;;      	 49c1e306             	shlq	$6, %r11
;;      	 4c01d8               	addq	%r11, %rax
;;      	 4883c418             	addq	$0x18, %rsp
;;      	 5d                   	popq	%rbp
;;      	 c3                   	retq	
;;   5e:	 0f0b                 	ud2	
