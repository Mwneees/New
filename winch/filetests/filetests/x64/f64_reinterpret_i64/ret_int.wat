;;! target = "x86_64"

(module
    (func (result i64)
        i64.const 1
        f64.reinterpret_i64
        drop
        i64.const 1
    )
)
;;      	 55                   	pushq	%rbp
;;      	 4889e5               	movq	%rsp, %rbp
;;      	 4c8b5f08             	movq	8(%rdi), %r11
;;      	 4d8b1b               	movq	(%r11), %r11
;;      	 4981c310000000       	addq	$0x10, %r11
;;      	 4939e3               	cmpq	%rsp, %r11
;;      	 0f8729000000         	ja	0x44
;;   1b:	 4989fe               	movq	%rdi, %r14
;;      	 4883ec10             	subq	$0x10, %rsp
;;      	 48897c2408           	movq	%rdi, 8(%rsp)
;;      	 48893424             	movq	%rsi, (%rsp)
;;      	 48c7c001000000       	movq	$1, %rax
;;      	 66480f6ec0           	movq	%rax, %xmm0
;;      	 48c7c001000000       	movq	$1, %rax
;;      	 4883c410             	addq	$0x10, %rsp
;;      	 5d                   	popq	%rbp
;;      	 c3                   	retq	
;;   44:	 0f0b                 	ud2	
