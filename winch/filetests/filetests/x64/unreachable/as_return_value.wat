;;! target = "x86_64"

(module
  (func (export "as-return-value") (result i64)
    (return (unreachable))
  )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4883ec08             	sub	rsp, 8
;;      	 4d8b5e08             	mov	r11, qword ptr [r14 + 8]
;;      	 4d8b1b               	mov	r11, qword ptr [r11]
;;      	 4939e3               	cmp	r11, rsp
;;      	 0f870c000000         	ja	0x24
;;   18:	 4c893424             	mov	qword ptr [rsp], r14
;;      	 0f0b                 	ud2	
;;      	 4883c408             	add	rsp, 8
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;   24:	 0f0b                 	ud2	
