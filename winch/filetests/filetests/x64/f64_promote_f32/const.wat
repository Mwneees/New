;;! target = "x86_64"

(module
    (func (result f64)
        (f32.const 1.0)
        (f64.promote_f32)
    )
)
;;      	 55                   	push	rbp
;;      	 4889e5               	mov	rbp, rsp
;;      	 4883ec08             	sub	rsp, 8
;;      	 4c893424             	mov	qword ptr [rsp], r14
;;      	 f30f10050c000000     	movss	xmm0, dword ptr [rip + 0xc]
;;      	 f30f5ac0             	cvtss2sd	xmm0, xmm0
;;      	 4883c408             	add	rsp, 8
;;      	 5d                   	pop	rbp
;;      	 c3                   	ret	
;;   1e:	 0000                 	add	byte ptr [rax], al
;;   20:	 0000                 	add	byte ptr [rax], al
