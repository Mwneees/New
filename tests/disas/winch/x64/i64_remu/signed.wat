;;! target = "x86_64"
;;! test = "winch"

(module
    (func (result i64)
	(i64.const -1)
	(i64.const -1)
	(i64.rem_u)
    )
)
;;      	 55                   	pushq	%rbp
;;      	 4889e5               	movq	%rsp, %rbp
;;      	 4c8b5f08             	movq	8(%rdi), %r11
;;      	 4d8b1b               	movq	(%r11), %r11
;;      	 4981c310000000       	addq	$0x10, %r11
;;      	 4939e3               	cmpq	%rsp, %r11
;;      	 0f872d000000         	ja	0x48
;;   1b:	 4989fe               	movq	%rdi, %r14
;;      	 4883ec10             	subq	$0x10, %rsp
;;      	 48897c2408           	movq	%rdi, 8(%rsp)
;;      	 48893424             	movq	%rsi, (%rsp)
;;      	 48c7c1ffffffff       	movq	$18446744073709551615, %rcx
;;      	 48c7c0ffffffff       	movq	$18446744073709551615, %rax
;;      	 4831d2               	xorq	%rdx, %rdx
;;      	 48f7f1               	divq	%rcx
;;      	 4889d0               	movq	%rdx, %rax
;;      	 4883c410             	addq	$0x10, %rsp
;;      	 5d                   	popq	%rbp
;;      	 c3                   	retq	
;;   48:	 0f0b                 	ud2	
