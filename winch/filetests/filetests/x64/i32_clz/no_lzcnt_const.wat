;;! target = "x86_64"

(module
    (func (result i32)
        (i32.const 1)
        (i32.clz)
    )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4d8b5e08             	mov	r11, qword ptr [r14 + 8]
;;      	 4d8b1b               	mov	r11, qword ptr [r11]
;;      	 4981c308000000       	add	r11, 8
;;      	 4939e3               	cmp	r11, rsp
;;      	 0f8728000000         	ja	0x43
;;   1b:	 4883ec08             	sub	rsp, 8
;;      	 4c893424             	mov	qword ptr [rsp], r14
;;      	 b801000000           	mov	eax, 1
;;      	 0fbdc0               	bsr	eax, eax
;;      	 41bb00000000         	mov	r11d, 0
;;      	 410f95c3             	setne	r11b
;;      	 f7d8                 	neg	eax
;;      	 83c020               	add	eax, 0x20
;;      	 4429d8               	sub	eax, r11d
;;      	 4883c408             	add	rsp, 8
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;   43:	 0f0b                 	ud2	
