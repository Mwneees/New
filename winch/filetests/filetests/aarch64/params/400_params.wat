;;! target = "aarch64"

(module
  (type (;0;) (func (param
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
    i32 i32 i32 i32 i32 i32 i32 i32 i32 i32
  )

    (result i32)
  ))
  (func (export "x") (type 0) local.get 0)
)
;;      	 fd7bbfa9             	stp	x29, x30, [sp, #-0x10]!
;;      	 fd030091             	mov	x29, sp
;;      	 fc030091             	mov	x28, sp
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 938300f8             	stur	x19, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 948300f8             	stur	x20, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 958300f8             	stur	x21, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 968300f8             	stur	x22, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 978300f8             	stur	x23, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 988300f8             	stur	x24, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 998300f8             	stur	x25, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 9a8300f8             	stur	x26, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 9b8300f8             	stur	x27, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 9c8300f8             	stur	x28, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 888300f8             	stur	x8, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 898300f8             	stur	x9, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8a8300f8             	stur	x10, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8b8300f8             	stur	x11, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8c8300f8             	stur	x12, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8d8300f8             	stur	x13, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8e8300f8             	stur	x14, [x28, #8]
;;      	 ff2300d1             	sub	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8f8300f8             	stur	x15, [x28, #8]
;;      	 e90300aa             	mov	x9, x0
;;      	 ffa300d1             	sub	sp, sp, #0x28
;;      	 fc030091             	mov	x28, sp
;;      	 80030bf8             	stur	x0, [x28, #0xb0]
;;      	 81830af8             	stur	x1, [x28, #0xa8]
;;      	 82430ab8             	stur	w2, [x28, #0xa4]
;;      	 83030ab8             	stur	w3, [x28, #0xa0]
;;      	 84c309b8             	stur	w4, [x28, #0x9c]
;;      	 858309b8             	stur	w5, [x28, #0x98]
;;      	 864309b8             	stur	w6, [x28, #0x94]
;;      	 870309b8             	stur	w7, [x28, #0x90]
;;      	 80434ab8             	ldur	w0, [x28, #0xa4]
;;      	 ffa30091             	add	sp, sp, #0x28
;;      	 fc030091             	mov	x28, sp
;;      	 8f8340f8             	ldur	x15, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8e8340f8             	ldur	x14, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8d8340f8             	ldur	x13, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8c8340f8             	ldur	x12, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8b8340f8             	ldur	x11, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 8a8340f8             	ldur	x10, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 898340f8             	ldur	x9, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 888340f8             	ldur	x8, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 9c8340f8             	ldur	x28, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 9b8340f8             	ldur	x27, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 9a8340f8             	ldur	x26, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 998340f8             	ldur	x25, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 988340f8             	ldur	x24, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 978340f8             	ldur	x23, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 968340f8             	ldur	x22, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 958340f8             	ldur	x21, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 948340f8             	ldur	x20, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 938340f8             	ldur	x19, [x28, #8]
;;      	 ff230091             	add	sp, sp, #8
;;      	 fc030091             	mov	x28, sp
;;      	 fd7bc1a8             	ldp	x29, x30, [sp], #0x10
;;      	 c0035fd6             	ret	
