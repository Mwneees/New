;;! target = "x86_64"
(module
    (func (result i64)
	(i64.const 0x7fffffffffffffff)
	(i64.const -1)
	(i64.sub)
    )
)
;;    0:	 55                   	push	rbp
;;    1:	 4889e5               	mov	rbp, rsp
;;    4:	 4883ec08             	sub	rsp, 8
;;    8:	 4c893424             	mov	qword ptr [rsp], r14
;;    c:	 48b8ffffffffffffff7f 	
;; 				movabs	rax, 0x7fffffffffffffff
;;   16:	 4883e8ff             	sub	rax, -1
;;   1a:	 4883c408             	add	rsp, 8
;;   1e:	 5d                   	pop	rbp
;;   1f:	 c3                   	ret	
