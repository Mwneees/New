;;! target = "x86_64"
;;! flags = ["has_lzcnt"]

(module
    (func (result i32)
        (i32.const 1)
        (i32.clz)
    )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4883ec08             	sub	rsp, 8
;;      	 4c893424             	mov	qword ptr [rsp], r14
;;      	 b801000000           	mov	eax, 1
;;      	 f30fbdc0             	lzcnt	eax, eax
;;      	 4883c408             	add	rsp, 8
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
