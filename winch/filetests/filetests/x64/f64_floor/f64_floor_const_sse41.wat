;;! target = "x86_64"
;;! flags = ["has_sse41"]

(module
    (func (result f64)
        (f64.const -1.32)
        (f64.floor)
    )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4d8b5e08             	mov	r11, qword ptr [r14 + 8]
;;      	 4d8b1b               	mov	r11, qword ptr [r11]
;;      	 4981c308000000       	add	r11, 8
;;      	 4939e3               	cmp	r11, rsp
;;      	 0f871c000000         	ja	0x37
;;   1b:	 4883ec08             	sub	rsp, 8
;;      	 4c893424             	mov	qword ptr [rsp], r14
;;      	 f20f100515000000     	movsd	xmm0, qword ptr [rip + 0x15]
;;      	 660f3a0bc001         	roundsd	xmm0, xmm0, 1
;;      	 4883c408             	add	rsp, 8
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;   37:	 0f0b                 	ud2	
;;   39:	 0000                 	add	byte ptr [rax], al
;;   3b:	 0000                 	add	byte ptr [rax], al
;;   3d:	 0000                 	add	byte ptr [rax], al
;;   3f:	 001f                 	add	byte ptr [rdi], bl
;;   41:	 85eb                 	test	ebx, ebp
;;   43:	 51                   	push	rcx
