;;! target = "x86_64"

(module
    (func (result i32)
        (local $foo i32)
        (local $bar i32)

        (i32.const 2)
        (local.set $foo)
        (i32.const 3)
        (local.set $bar)

        (local.get $foo)
        (local.get $bar)
        (i32.le_s)
    )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4c8b5f08             	mov	r11, qword ptr [rdi + 8]
;;      	 4d8b1b               	mov	r11, qword ptr [r11]
;;      	 4981c348000000       	add	r11, 0x48
;;      	 4939e3               	cmp	r11, rsp
;;      	 0f877f000000         	ja	0x9a
;;   1b:	 4883ec30             	sub	rsp, 0x30
;;      	 48891c24             	mov	qword ptr [rsp], rbx
;;      	 4c89642408           	mov	qword ptr [rsp + 8], r12
;;      	 4c896c2410           	mov	qword ptr [rsp + 0x10], r13
;;      	 4c89742418           	mov	qword ptr [rsp + 0x18], r14
;;      	 4c897c2420           	mov	qword ptr [rsp + 0x20], r15
;;      	 4989fe               	mov	r14, rdi
;;      	 4883ec18             	sub	rsp, 0x18
;;      	 48897c2440           	mov	qword ptr [rsp + 0x40], rdi
;;      	 4889742438           	mov	qword ptr [rsp + 0x38], rsi
;;      	 48c744243000000000   	
;; 				mov	qword ptr [rsp + 0x30], 0
;;      	 b802000000           	mov	eax, 2
;;      	 89442434             	mov	dword ptr [rsp + 0x34], eax
;;      	 b803000000           	mov	eax, 3
;;      	 89442430             	mov	dword ptr [rsp + 0x30], eax
;;      	 8b442430             	mov	eax, dword ptr [rsp + 0x30]
;;      	 8b4c2434             	mov	ecx, dword ptr [rsp + 0x34]
;;      	 39c1                 	cmp	ecx, eax
;;      	 b900000000           	mov	ecx, 0
;;      	 400f9ec1             	setle	cl
;;      	 89c8                 	mov	eax, ecx
;;      	 4883c418             	add	rsp, 0x18
;;      	 488b1c24             	mov	rbx, qword ptr [rsp]
;;      	 4c8b642408           	mov	r12, qword ptr [rsp + 8]
;;      	 4c8b6c2410           	mov	r13, qword ptr [rsp + 0x10]
;;      	 4c8b742418           	mov	r14, qword ptr [rsp + 0x18]
;;      	 4c8b7c2420           	mov	r15, qword ptr [rsp + 0x20]
;;      	 4883c430             	add	rsp, 0x30
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;   9a:	 0f0b                 	ud2	
