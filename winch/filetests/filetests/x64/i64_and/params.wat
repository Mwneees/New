;;! target = "x86_64"

(module
    (func (param i64) (param i64) (result i64)
        (local.get 0)
        (local.get 1)
        (i64.and)
    )
)
;;    0:	 55                   	push	rbp
;;    1:	 4889e5               	mov	rbp, rsp
;;    4:	 4883ec18             	sub	rsp, 0x18
;;    8:	 48897c2410           	mov	qword ptr [rsp + 0x10], rdi
;;    d:	 4889742408           	mov	qword ptr [rsp + 8], rsi
;;   12:	 4c893424             	mov	qword ptr [rsp], r14
;;   16:	 488b442408           	mov	rax, qword ptr [rsp + 8]
;;   1b:	 488b4c2410           	mov	rcx, qword ptr [rsp + 0x10]
;;   20:	 4821c1               	and	rcx, rax
;;   23:	 4889c8               	mov	rax, rcx
;;   26:	 4883c418             	add	rsp, 0x18
;;   2a:	 5d                   	pop	rbp
;;   2b:	 c3                   	ret	
