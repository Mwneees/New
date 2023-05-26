;;! target = "x86_64"
;;!
;;! settings = ['enable_heap_access_spectre_mitigation=false']
;;!
;;! compile = true
;;!
;;! [globals.vmctx]
;;! type = "i64"
;;! vmctx = true
;;!
;;! [globals.heap_base]
;;! type = "i64"
;;! load = { base = "vmctx", offset = 0, readonly = true }
;;!
;;! [globals.heap_bound]
;;! type = "i64"
;;! load = { base = "vmctx", offset = 8, readonly = true }
;;!
;;! [[heaps]]
;;! base = "heap_base"
;;! min_size = 0x10000
;;! offset_guard_size = 0
;;! index_type = "i32"
;;! style = { kind = "dynamic", bound = "heap_bound" }

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load offset=0xffff0000))

;; function u0:0:
;;   push rbp
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mov rbp, rsp
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   mov r8d, edi
;;   mov r11, r8
;;   add r11, r11, const(0)
;;   jb #trap=heap_oob
;;   mov rdi, qword ptr [rdx + 0x8]
;;   cmp r11, rdi
;;   jnbe label3; j label1
;; block1:
;;   add r8, r8, qword ptr [rdx + 0x0]
;;   mov eax, 0xffff0000
;;   mov dword ptr [r8 + rax], esi
;;   jmp label2
;; block2:
;;   mov rsp, rbp
;;   pop rbp
;;   ret
;; block3:
;;   ud2 heap_oob
;;
;; function u0:1:
;;   push rbp
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mov rbp, rsp
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   mov r8d, edi
;;   mov r11, r8
;;   add r11, r11, const(0)
;;   jb #trap=heap_oob
;;   mov rdi, qword ptr [rsi + 0x8]
;;   cmp r11, rdi
;;   jnbe label3; j label1
;; block1:
;;   add r8, r8, qword ptr [rsi + 0x0]
;;   mov eax, 0xffff0000
;;   mov eax, dword ptr [r8 + rax]
;;   jmp label2
;; block2:
;;   mov rsp, rbp
;;   pop rbp
;;   ret
;; block3:
;;   ud2 heap_oob
