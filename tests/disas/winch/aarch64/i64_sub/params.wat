;;! target = "aarch64"
;;! test = "winch"

(module
    (func (param i64) (param i64) (result i64)
	(local.get 0)
	(local.get 1)
	(i64.sub)
    )
)
;;      	 fd7bbfa9             	stp	x29, x30, [sp, #-0x10]!
;;      	 fd030091             	mov	x29, sp
;;      	 fc030091             	mov	x28, sp
;;      	 e90300aa             	mov	x9, x0
;;      	 ff8300d1             	sub	sp, sp, #0x20
;;      	 fc030091             	mov	x28, sp
;;      	 808301f8             	stur	x0, [x28, #0x18]
;;      	 810301f8             	stur	x1, [x28, #0x10]
;;      	 828300f8             	stur	x2, [x28, #8]
;;      	 830300f8             	stur	x3, [x28]
;;      	 800340f8             	ldur	x0, [x28]
;;      	 818340f8             	ldur	x1, [x28, #8]
;;      	 216020cb             	sub	x1, x1, x0, uxtx
;;      	 e00301aa             	mov	x0, x1
;;      	 ff830091             	add	sp, sp, #0x20
;;      	 fc030091             	mov	x28, sp
;;      	 fd7bc1a8             	ldp	x29, x30, [sp], #0x10
;;      	 c0035fd6             	ret	
