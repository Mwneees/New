;;! target = "x86_64"

(module 
  (func $dummy)
  (func (export "as-block-mid") (result i32)
    (block (result i32) (call $dummy) (unreachable) (i32.const 2))
  )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4883ec08             	sub	rsp, 8
;;      	 4c893424             	mov	qword ptr [rsp], r14
;;      	 4883c408             	add	rsp, 8
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4883ec08             	sub	rsp, 8
;;      	 4c893424             	mov	qword ptr [rsp], r14
;;      	 4883ec08             	sub	rsp, 8
;;      	 e800000000           	call	0x15
;;      	 4883c408             	add	rsp, 8
;;      	 0f0b                 	ud2	
;;      	 4883c408             	add	rsp, 8
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
