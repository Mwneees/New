;;! target = "x86_64"

(module
    (func (result f64)
        (local $foo f64)  
        (local $bar f64)

        (f64.const 1.1)
        (local.set $foo)

        (f64.const 2.2)
        (local.set $bar)

        (local.get $foo)
        (local.get $bar)
        f64.sub
    )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4c8b5f08             	mov	r11, qword ptr [rdi + 8]
;;      	 4d8b1b               	mov	r11, qword ptr [r11]
;;      	 4981c350000000       	add	r11, 0x50
;;      	 4939e3               	cmp	r11, rsp
;;      	 0f878c000000         	ja	0xa7
;;   1b:	 4883ec30             	sub	rsp, 0x30
;;      	 48891c24             	mov	qword ptr [rsp], rbx
;;      	 4c89642408           	mov	qword ptr [rsp + 8], r12
;;      	 4c896c2410           	mov	qword ptr [rsp + 0x10], r13
;;      	 4c89742418           	mov	qword ptr [rsp + 0x18], r14
;;      	 4c897c2420           	mov	qword ptr [rsp + 0x20], r15
;;      	 4989fe               	mov	r14, rdi
;;      	 4883ec20             	sub	rsp, 0x20
;;      	 48897c2448           	mov	qword ptr [rsp + 0x48], rdi
;;      	 4889742440           	mov	qword ptr [rsp + 0x40], rsi
;;      	 4531db               	xor	r11d, r11d
;;      	 4c895c2438           	mov	qword ptr [rsp + 0x38], r11
;;      	 4c895c2430           	mov	qword ptr [rsp + 0x30], r11
;;      	 f20f100553000000     	movsd	xmm0, qword ptr [rip + 0x53]
;;      	 f20f11442438         	movsd	qword ptr [rsp + 0x38], xmm0
;;      	 f20f10054d000000     	movsd	xmm0, qword ptr [rip + 0x4d]
;;      	 f20f11442430         	movsd	qword ptr [rsp + 0x30], xmm0
;;      	 f20f10442430         	movsd	xmm0, qword ptr [rsp + 0x30]
;;      	 f20f104c2438         	movsd	xmm1, qword ptr [rsp + 0x38]
;;      	 f20f5cc8             	subsd	xmm1, xmm0
;;      	 660f28c1             	movapd	xmm0, xmm1
;;      	 4883c420             	add	rsp, 0x20
;;      	 488b1c24             	mov	rbx, qword ptr [rsp]
;;      	 4c8b642408           	mov	r12, qword ptr [rsp + 8]
;;      	 4c8b6c2410           	mov	r13, qword ptr [rsp + 0x10]
;;      	 4c8b742418           	mov	r14, qword ptr [rsp + 0x18]
;;      	 4c8b7c2420           	mov	r15, qword ptr [rsp + 0x20]
;;      	 4883c430             	add	rsp, 0x30
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;   a7:	 0f0b                 	ud2	
;;   a9:	 0000                 	add	byte ptr [rax], al
;;   ab:	 0000                 	add	byte ptr [rax], al
;;   ad:	 0000                 	add	byte ptr [rax], al
;;   af:	 009a99999999         	add	byte ptr [rdx - 0x66666667], bl
;;   b5:	 99                   	cdq	
;;   b6:	 f1                   	int1	
