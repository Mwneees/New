;;! target = "x86_64"

(module
    (func (result i32)
	(i32.const 10)
	(i32.const 20)
	(i32.add)
    )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4883ec08             	sub	rsp, 8
;;      	 4d8b5e08             	mov	r11, qword ptr [r14 + 8]
;;      	 4d8b1b               	mov	r11, qword ptr [r11]
;;      	 4939e3               	cmp	r11, rsp
;;      	 0f8712000000         	ja	0x2a
;;   18:	 4c893424             	mov	qword ptr [rsp], r14
;;      	 b80a000000           	mov	eax, 0xa
;;      	 83c014               	add	eax, 0x14
;;      	 4883c408             	add	rsp, 8
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;   2a:	 0f0b                 	ud2	
