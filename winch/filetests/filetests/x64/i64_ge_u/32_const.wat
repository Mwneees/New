;;! target = "x86_64"

(module
    (func (result i32)
        (i64.const 2)
        (i64.const 3)
        (i64.ge_u)
    )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4d8b5e08             	mov	r11, qword ptr [r14 + 8]
;;      	 4d8b1b               	mov	r11, qword ptr [r11]
;;      	 4981c308000000       	add	r11, 8
;;      	 4939e3               	cmp	r11, rsp
;;      	 0f8722000000         	ja	0x3d
;;   1b:	 4883ec08             	sub	rsp, 8
;;      	 4c893424             	mov	qword ptr [rsp], r14
;;      	 48c7c002000000       	mov	rax, 2
;;      	 4883f803             	cmp	rax, 3
;;      	 b800000000           	mov	eax, 0
;;      	 400f93c0             	setae	al
;;      	 4883c408             	add	rsp, 8
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;   3d:	 0f0b                 	ud2	
