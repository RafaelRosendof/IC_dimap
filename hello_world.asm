section .data
  hello_msg db 'Hello , Figas! ', 0

section .text
  global _start

_start:
  ;Printando a mensagem

  mov rax , 1        ;chamando o syscall pra escrever
  mov rdi, 1     ;stdout
  mov rsi, hello_msg ;endereço da mensagem

  mov rdx, 13 ;tamanho 
  syscall

  ;Saindo do programa

  mov rax, 60 ;syscall numero para sair 
  xor rdi , rdi   ;return code 0
  syscall
