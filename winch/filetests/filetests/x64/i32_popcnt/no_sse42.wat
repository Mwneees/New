;;! target = "x86_64"
;;! flags = ["has_popcnt"]

(module
    (func (result i32)
      i32.const 3
      i32.popcnt
    )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4c8b5f08             	mov	r11, qword ptr [rdi + 8]
;;      	 4d8b1b               	mov	r11, qword ptr [r11]
;;      	 4981c340000000       	add	r11, 0x40
;;      	 4939e3               	cmp	r11, rsp
;;      	 0f878a000000         	ja	0xa5
;;   1b:	 4883ec30             	sub	rsp, 0x30
;;      	 48891c24             	mov	qword ptr [rsp], rbx
;;      	 4c89642408           	mov	qword ptr [rsp + 8], r12
;;      	 4c896c2410           	mov	qword ptr [rsp + 0x10], r13
;;      	 4c89742418           	mov	qword ptr [rsp + 0x18], r14
;;      	 4c897c2420           	mov	qword ptr [rsp + 0x20], r15
;;      	 4989fe               	mov	r14, rdi
;;      	 4883ec10             	sub	rsp, 0x10
;;      	 48897c2438           	mov	qword ptr [rsp + 0x38], rdi
;;      	 4889742430           	mov	qword ptr [rsp + 0x30], rsi
;;      	 b803000000           	mov	eax, 3
;;      	 89c1                 	mov	ecx, eax
;;      	 c1e801               	shr	eax, 1
;;      	 81e055555555         	and	eax, 0x55555555
;;      	 29c1                 	sub	ecx, eax
;;      	 89c8                 	mov	eax, ecx
;;      	 41bb33333333         	mov	r11d, 0x33333333
;;      	 4421d8               	and	eax, r11d
;;      	 c1e902               	shr	ecx, 2
;;      	 4421d9               	and	ecx, r11d
;;      	 01c1                 	add	ecx, eax
;;      	 89c8                 	mov	eax, ecx
;;      	 c1e804               	shr	eax, 4
;;      	 01c8                 	add	eax, ecx
;;      	 81e00f0f0f0f         	and	eax, 0xf0f0f0f
;;      	 69c001010101         	imul	eax, eax, 0x1010101
;;      	 c1e818               	shr	eax, 0x18
;;      	 4883c410             	add	rsp, 0x10
;;      	 488b1c24             	mov	rbx, qword ptr [rsp]
;;      	 4c8b642408           	mov	r12, qword ptr [rsp + 8]
;;      	 4c8b6c2410           	mov	r13, qword ptr [rsp + 0x10]
;;      	 4c8b742418           	mov	r14, qword ptr [rsp + 0x18]
;;      	 4c8b7c2420           	mov	r15, qword ptr [rsp + 0x20]
;;      	 4883c430             	add	rsp, 0x30
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;   a5:	 0f0b                 	ud2	
