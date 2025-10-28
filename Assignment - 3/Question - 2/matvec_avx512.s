	.file	"matvec_avx512.c"
	.text
	.p2align 4
	.globl	get_time_millis
	.type	get_time_millis, @function
get_time_millis:
.LFB5495:
	.cfi_startproc
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movl	$1, %edi
	movq	%rsp, %rsi
	call	clock_gettime
	vxorps	%xmm1, %xmm1, %xmm1
	vcvtsi2sdq	(%rsp), %xmm1, %xmm0
	vcvtsi2sdq	8(%rsp), %xmm1, %xmm1
	vdivsd	.LC0(%rip), %xmm1, %xmm1
	vfmadd132sd	.LC1(%rip), %xmm1, %xmm0
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE5495:
	.size	get_time_millis, .-get_time_millis
	.p2align 4
	.globl	matvec_naive
	.type	matvec_naive, @function
matvec_naive:
.LFB5496:
	.cfi_startproc
	testl	%ecx, %ecx
	jle	.L19
	leal	-1(%rcx), %eax
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	%r8d, %r10d
	movq	%rdx, %r9
	leaq	4(%rdx,%rax,4), %r11
	movl	%r8d, %eax
	andl	$-16, %r10d
	xorl	%edx, %edx
	shrl	$4, %eax
	vxorps	%xmm3, %xmm3, %xmm3
	leal	-1(%rax), %ecx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	addq	$1, %rcx
	pushq	%r13
	salq	$6, %rcx
	pushq	%r12
	pushq	%rbx
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.cfi_offset 12, -40
	.cfi_offset 3, -48
	leal	-1(%r8), %ebx
	.p2align 4,,10
	.p2align 3
.L6:
	vmovaps	%xmm3, %xmm0
	testl	%r8d, %r8d
	jle	.L13
	cmpl	$14, %ebx
	jbe	.L14
	movslq	%edx, %rax
	vmovaps	%xmm3, %xmm0
	leaq	(%rdi,%rax,4), %r12
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L8:
	vmovups	(%r12,%rax), %zmm6
	vmulps	(%rsi,%rax), %zmm6, %zmm1
	addq	$64, %rax
	vaddss	%xmm1, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm5
	vextractf128	$0x1, %ymm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm4
	vaddss	%xmm5, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm5
	vextracti64x4	$0x1, %zmm1, %ymm1
	vaddss	%xmm5, %xmm0, %xmm0
	vaddss	%xmm4, %xmm0, %xmm0
	vshufps	$85, %xmm2, %xmm2, %xmm4
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm4, %xmm0, %xmm0
	vunpckhps	%xmm2, %xmm2, %xmm4
	vshufps	$255, %xmm2, %xmm2, %xmm2
	vaddss	%xmm4, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm4
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm4, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm4
	vextractf128	$0x1, %ymm1, %xmm1
	vaddss	%xmm4, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	%rcx, %rax
	jne	.L8
	cmpl	%r8d, %r10d
	je	.L13
	movl	%r10d, %r12d
	movl	%r10d, %eax
.L7:
	movl	%r8d, %r14d
	subl	%r12d, %r14d
	leal	-1(%r14), %r13d
	cmpl	$6, %r13d
	jbe	.L10
	movslq	%edx, %r13
	vmovups	(%rsi,%r12,4), %ymm1
	addq	%r12, %r13
	movl	%r14d, %r12d
	vmulps	(%rdi,%r13,4), %ymm1, %ymm1
	andl	$-8, %r12d
	addl	%r12d, %eax
	vaddss	%xmm1, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm4
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm4, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm4
	vextractf128	$0x1, %ymm1, %xmm1
	vaddss	%xmm4, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpl	%r12d, %r14d
	je	.L13
.L10:
	leal	(%rdx,%rax), %r12d
	movslq	%eax, %r14
	movslq	%r12d, %r12
	leaq	0(,%r14,4), %r13
	vmovss	(%rdi,%r12,4), %xmm7
	leal	1(%rax), %r12d
	vfmadd231ss	(%rsi,%r14,4), %xmm7, %xmm0
	cmpl	%r12d, %r8d
	jle	.L13
	addl	%edx, %r12d
	movslq	%r12d, %r12
	vmovss	(%rdi,%r12,4), %xmm7
	leal	2(%rax), %r12d
	vfmadd231ss	4(%rsi,%r13), %xmm7, %xmm0
	cmpl	%r12d, %r8d
	jle	.L13
	addl	%edx, %r12d
	movslq	%r12d, %r12
	vmovss	(%rdi,%r12,4), %xmm7
	leal	3(%rax), %r12d
	vfmadd231ss	8(%rsi,%r13), %xmm7, %xmm0
	cmpl	%r12d, %r8d
	jle	.L13
	addl	%edx, %r12d
	movslq	%r12d, %r12
	vmovss	(%rdi,%r12,4), %xmm7
	leal	4(%rax), %r12d
	vfmadd231ss	12(%rsi,%r13), %xmm7, %xmm0
	cmpl	%r12d, %r8d
	jle	.L13
	addl	%edx, %r12d
	movslq	%r12d, %r12
	vmovss	(%rdi,%r12,4), %xmm7
	leal	5(%rax), %r12d
	vfmadd231ss	16(%rsi,%r13), %xmm7, %xmm0
	cmpl	%r12d, %r8d
	jle	.L13
	addl	%edx, %r12d
	addl	$6, %eax
	movslq	%r12d, %r12
	vmovss	(%rdi,%r12,4), %xmm7
	vfmadd231ss	20(%rsi,%r13), %xmm7, %xmm0
	cmpl	%eax, %r8d
	jle	.L13
	addl	%edx, %eax
	cltq
	vmovss	(%rdi,%rax,4), %xmm7
	vfmadd231ss	24(%rsi,%r13), %xmm7, %xmm0
.L13:
	vmovss	%xmm0, (%r9)
	addq	$4, %r9
	addl	%r8d, %edx
	cmpq	%r11, %r9
	jne	.L6
	vzeroupper
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L14:
	.cfi_restore_state
	xorl	%r12d, %r12d
	xorl	%eax, %eax
	vmovaps	%xmm3, %xmm0
	jmp	.L7
.L19:
	.cfi_def_cfa 7, 8
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	ret
	.cfi_endproc
.LFE5496:
	.size	matvec_naive, .-matvec_naive
	.p2align 4
	.globl	matvec_avx512
	.type	matvec_avx512, @function
matvec_avx512:
.LFB5497:
	.cfi_startproc
	testl	%ecx, %ecx
	jle	.L38
	leal	-16(%r8), %r11d
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %r9
	leal	-1(%rcx), %eax
	shrl	$4, %r11d
	movq	%rdx, %r10
	vxorps	%xmm3, %xmm3, %xmm3
	xorl	%ecx, %ecx
	leal	1(%r11), %edi
	movq	%rdi, %r11
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	salq	$6, %rdi
	pushq	%r14
	sall	$4, %r11d
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.cfi_offset 12, -40
	.cfi_offset 3, -48
	leaq	4(%rdx,%rax,4), %rbx
	.p2align 4,,10
	.p2align 3
.L24:
	vmovaps	%xmm3, %xmm0
	xorl	%r12d, %r12d
	cmpl	$15, %r8d
	jle	.L32
	movslq	%ecx, %rax
	vxorps	%xmm1, %xmm1, %xmm1
	leaq	(%r9,%rax,4), %rdx
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L25:
	vmovups	(%rdx,%rax), %zmm6
	vfmadd231ps	(%rsi,%rax), %zmm6, %zmm1
	addq	$64, %rax
	cmpq	%rax, %rdi
	jne	.L25
	vaddss	%xmm3, %xmm1, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm5
	vextractf128	$0x1, %ymm1, %xmm2
	movl	%r11d, %r12d
	vshufps	$255, %xmm1, %xmm1, %xmm4
	vaddss	%xmm5, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm5
	vextracti64x4	$0x1, %zmm1, %ymm1
	vaddss	%xmm5, %xmm0, %xmm0
	vaddss	%xmm4, %xmm0, %xmm0
	vshufps	$85, %xmm2, %xmm2, %xmm4
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm4, %xmm0, %xmm0
	vunpckhps	%xmm2, %xmm2, %xmm4
	vshufps	$255, %xmm2, %xmm2, %xmm2
	vaddss	%xmm4, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm4
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm4, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm4
	vextractf128	$0x1, %ymm1, %xmm1
	vaddss	%xmm4, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
.L32:
	cmpl	%r12d, %r8d
	jle	.L26
	movl	%r8d, %r13d
	subl	%r12d, %r13d
	leal	-1(%r13), %eax
	cmpl	$14, %eax
	jbe	.L33
	movslq	%r12d, %rax
	movslq	%ecx, %rdx
	addq	%rax, %rdx
	vmovups	(%rsi,%rax,4), %zmm1
	movl	%r13d, %eax
	vmulps	(%r9,%rdx,4), %zmm1, %zmm1
	andl	$-16, %eax
	leal	(%rax,%r12), %edx
	vaddss	%xmm0, %xmm1, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm5
	vshufps	$255, %xmm1, %xmm1, %xmm4
	vaddss	%xmm0, %xmm5, %xmm5
	vunpckhps	%xmm1, %xmm1, %xmm0
	vaddss	%xmm5, %xmm0, %xmm0
	vaddss	%xmm0, %xmm4, %xmm4
	vextractf128	$0x1, %ymm1, %xmm0
	vextracti64x4	$0x1, %zmm1, %ymm1
	vaddss	%xmm4, %xmm0, %xmm2
	vshufps	$85, %xmm0, %xmm0, %xmm4
	vaddss	%xmm2, %xmm4, %xmm4
	vunpckhps	%xmm0, %xmm0, %xmm2
	vshufps	$255, %xmm0, %xmm0, %xmm0
	vaddss	%xmm4, %xmm2, %xmm2
	vshufps	$85, %xmm1, %xmm1, %xmm4
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm0, %xmm1, %xmm0
	vaddss	%xmm0, %xmm4, %xmm4
	vunpckhps	%xmm1, %xmm1, %xmm0
	vextractf128	$0x1, %ymm1, %xmm1
	vaddss	%xmm4, %xmm0, %xmm0
	vaddss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm2, %xmm1, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vaddss	%xmm0, %xmm2, %xmm2
	vunpckhps	%xmm1, %xmm1, %xmm0
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpl	%r13d, %eax
	je	.L26
.L27:
	subl	%eax, %r13d
	leal	-1(%r13), %r14d
	cmpl	$6, %r14d
	jbe	.L29
	movslq	%r12d, %r14
	movslq	%ecx, %r12
	addq	%r14, %rax
	addq	%rax, %r12
	vmovups	(%rsi,%rax,4), %ymm1
	movl	%r13d, %eax
	vmulps	(%r9,%r12,4), %ymm1, %ymm1
	andl	$-8, %eax
	addl	%eax, %edx
	vaddss	%xmm0, %xmm1, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm4
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm0, %xmm4, %xmm4
	vunpckhps	%xmm1, %xmm1, %xmm0
	vextractf128	$0x1, %ymm1, %xmm1
	vaddss	%xmm4, %xmm0, %xmm0
	vaddss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm2, %xmm1, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vaddss	%xmm0, %xmm2, %xmm2
	vunpckhps	%xmm1, %xmm1, %xmm0
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpl	%r13d, %eax
	je	.L26
.L29:
	leal	(%rcx,%rdx), %eax
	movslq	%edx, %r13
	cltq
	leaq	0(,%r13,4), %r12
	vmovss	(%r9,%rax,4), %xmm7
	leal	1(%rdx), %eax
	vfmadd231ss	(%rsi,%r13,4), %xmm7, %xmm0
	cmpl	%eax, %r8d
	jle	.L26
	addl	%ecx, %eax
	vmovss	4(%rsi,%r12), %xmm7
	cltq
	vfmadd231ss	(%r9,%rax,4), %xmm7, %xmm0
	leal	2(%rdx), %eax
	cmpl	%eax, %r8d
	jle	.L26
	addl	%ecx, %eax
	vmovss	8(%rsi,%r12), %xmm7
	cltq
	vfmadd231ss	(%r9,%rax,4), %xmm7, %xmm0
	leal	3(%rdx), %eax
	cmpl	%eax, %r8d
	jle	.L26
	addl	%ecx, %eax
	vmovss	12(%rsi,%r12), %xmm7
	cltq
	vfmadd231ss	(%r9,%rax,4), %xmm7, %xmm0
	leal	4(%rdx), %eax
	cmpl	%r8d, %eax
	jge	.L26
	addl	%ecx, %eax
	cltq
	vmovss	(%r9,%rax,4), %xmm7
	leal	5(%rdx), %eax
	vfmadd231ss	16(%rsi,%r12), %xmm7, %xmm0
	cmpl	%eax, %r8d
	jle	.L26
	addl	%ecx, %eax
	addl	$6, %edx
	vmovss	20(%rsi,%r12), %xmm7
	cltq
	vfmadd231ss	(%r9,%rax,4), %xmm7, %xmm0
	cmpl	%edx, %r8d
	jle	.L26
	addl	%ecx, %edx
	vmovss	24(%rsi,%r12), %xmm7
	movslq	%edx, %rdx
	vfmadd231ss	(%r9,%rdx,4), %xmm7, %xmm0
.L26:
	vmovss	%xmm0, (%r10)
	addq	$4, %r10
	addl	%r8d, %ecx
	cmpq	%r10, %rbx
	jne	.L24
	vzeroupper
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L33:
	.cfi_restore_state
	movl	%r12d, %edx
	xorl	%eax, %eax
	jmp	.L27
.L38:
	.cfi_def_cfa 7, 8
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	ret
	.cfi_endproc
.LFE5497:
	.size	matvec_avx512, .-matvec_avx512
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC3:
	.string	"Usage: %s M N\n"
.LC6:
	.string	"Time naive (ms):    %f\n"
.LC7:
	.string	"Time AVX-512 (ms):  %f\n"
.LC8:
	.string	"Speedup:            %f\n"
.LC9:
	.string	"Max diff:           %f\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB5498:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-64, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	%rsi, %rbx
	subq	$64, %rsp
	cmpl	$2, %edi
	jle	.L72
	movq	8(%rsi), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol
	movq	16(%rbx), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	movq	%rax, %r14
	movl	%eax, -100(%rbp)
	call	strtol
	movq	%r14, -72(%rbp)
	movl	$64, %edi
	movq	%rax, %r15
	movq	%rax, -64(%rbp)
	movl	%eax, %ebx
	movl	%r14d, %eax
	imull	%r15d, %eax
	movslq	%eax, %rsi
	movq	%rsi, %r14
	salq	$2, %rsi
	call	aligned_alloc
	movslq	%r15d, %rsi
	movl	$64, %edi
	salq	$2, %rsi
	movq	%rax, %r13
	call	aligned_alloc
	movl	$64, %edi
	movq	%rax, %r12
	movslq	-72(%rbp), %rax
	leaq	0(,%rax,4), %r15
	movq	%r15, %rsi
	call	aligned_alloc
	movl	$64, %edi
	movq	%r15, %rsi
	movq	%rax, -88(%rbp)
	call	aligned_alloc
	xorl	%edi, %edi
	movq	%rax, -80(%rbp)
	call	srand
	testl	%r14d, %r14d
	jle	.L44
	vmovss	.LC4(%rip), %xmm4
	leal	-1(%r14), %eax
	movq	%r13, %r15
	leaq	4(%r13,%rax,4), %r14
	vmovss	%xmm4, -56(%rbp)
	.p2align 4,,10
	.p2align 3
.L45:
	call	rand
	vxorps	%xmm5, %xmm5, %xmm5
	addq	$4, %r15
	movslq	%eax, %rdx
	movl	%eax, %ecx
	imulq	$1374389535, %rdx, %rdx
	sarl	$31, %ecx
	sarq	$37, %rdx
	subl	%ecx, %edx
	imull	$100, %edx, %edx
	subl	%edx, %eax
	vcvtsi2ssl	%eax, %xmm5, %xmm0
	vdivss	-56(%rbp), %xmm0, %xmm0
	vmovss	%xmm0, -4(%r15)
	cmpq	%r14, %r15
	jne	.L45
.L44:
	movq	-64(%rbp), %rax
	testl	%eax, %eax
	jle	.L46
	vmovss	.LC4(%rip), %xmm4
	subl	$1, %eax
	movq	%r12, %r15
	leaq	4(%r12,%rax,4), %r14
	vmovss	%xmm4, -56(%rbp)
	.p2align 4,,10
	.p2align 3
.L47:
	call	rand
	vxorps	%xmm4, %xmm4, %xmm4
	addq	$4, %r15
	movslq	%eax, %rdx
	movl	%eax, %ecx
	imulq	$1374389535, %rdx, %rdx
	sarl	$31, %ecx
	sarq	$37, %rdx
	subl	%ecx, %edx
	imull	$100, %edx, %edx
	subl	%edx, %eax
	vcvtsi2ssl	%eax, %xmm4, %xmm0
	vdivss	-56(%rbp), %xmm0, %xmm0
	vmovss	%xmm0, -4(%r15)
	cmpq	%r14, %r15
	jne	.L47
.L46:
	xorl	%eax, %eax
	call	get_time_millis
	movq	-72(%rbp), %rsi
	vmovsd	%xmm0, -96(%rbp)
	testl	%esi, %esi
	jle	.L48
	movq	-64(%rbp), %rax
	movq	-88(%rbp), %rcx
	subl	$1, %esi
	xorl	%edx, %edx
	movq	%rsi, -72(%rbp)
	vxorps	%xmm4, %xmm4, %xmm4
	movl	%eax, %r10d
	leaq	4(%rcx,%rsi,4), %r9
	movq	%rcx, %r8
	movl	%eax, %r14d
	movq	%rax, %rcx
	subl	$1, %eax
	movl	%r10d, %r11d
	movl	%eax, -56(%rbp)
	movl	%r10d, %eax
	andl	$-16, %r11d
	shrl	$4, %eax
	leal	-1(%rax), %esi
	addq	$1, %rsi
	salq	$6, %rsi
	.p2align 4,,10
	.p2align 3
.L49:
	vmovaps	%xmm4, %xmm0
	testl	%ebx, %ebx
	jle	.L57
	cmpl	$14, -56(%rbp)
	jbe	.L64
	movslq	%edx, %rax
	vmovaps	%xmm4, %xmm0
	leaq	0(%r13,%rax,4), %rdi
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L51:
	vmovups	(%rdi,%rax), %zmm7
	vmulps	(%r12,%rax), %zmm7, %zmm2
	addq	$64, %rax
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$85, %xmm2, %xmm2, %xmm1
	vshufps	$255, %xmm2, %xmm2, %xmm6
	vaddss	%xmm1, %xmm0, %xmm0
	vunpckhps	%xmm2, %xmm2, %xmm1
	vaddss	%xmm1, %xmm0, %xmm0
	vextractf128	$0x1, %ymm2, %xmm1
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vaddss	%xmm6, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vextracti64x4	$0x1, %zmm2, %ymm1
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vextractf128	$0x1, %ymm1, %xmm1
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	%rax, %rsi
	jne	.L51
	cmpl	%r14d, %r11d
	je	.L57
	movl	%r11d, %edi
	movl	%r11d, %eax
.L50:
	movl	%r14d, %r15d
	subl	%edi, %r15d
	leal	-1(%r15), %r10d
	cmpl	$6, %r10d
	jbe	.L53
	movslq	%edx, %r10
	addq	%rdi, %r10
	vmovups	0(%r13,%r10,4), %ymm1
	vmulps	(%r12,%rdi,4), %ymm1, %ymm1
	movl	%r15d, %edi
	andl	$-8, %edi
	addl	%edi, %eax
	vaddss	%xmm1, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vextractf128	$0x1, %ymm1, %xmm1
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpl	%edi, %r15d
	je	.L57
.L53:
	leal	(%rdx,%rax), %edi
	movslq	%eax, %r10
	movslq	%edi, %rdi
	leaq	0(,%r10,4), %r15
	vmovss	0(%r13,%rdi,4), %xmm5
	leal	1(%rax), %edi
	vfmadd231ss	(%r12,%r10,4), %xmm5, %xmm0
	cmpl	%edi, %ebx
	jle	.L57
	addl	%edx, %edi
	vmovss	4(%r12,%r15), %xmm5
	movslq	%edi, %rdi
	vfmadd231ss	0(%r13,%rdi,4), %xmm5, %xmm0
	leal	2(%rax), %edi
	cmpl	%edi, %ebx
	jle	.L57
	addl	%edx, %edi
	movslq	%edi, %rdi
	vmovss	0(%r13,%rdi,4), %xmm7
	leal	3(%rax), %edi
	vfmadd231ss	8(%r12,%r15), %xmm7, %xmm0
	cmpl	%edi, %ebx
	jle	.L57
	addl	%edx, %edi
	movslq	%edi, %rdi
	vmovss	0(%r13,%rdi,4), %xmm5
	leal	4(%rax), %edi
	vfmadd231ss	12(%r12,%r15), %xmm5, %xmm0
	cmpl	%edi, %ebx
	jle	.L57
	addl	%edx, %edi
	movslq	%edi, %rdi
	vmovss	0(%r13,%rdi,4), %xmm7
	leal	5(%rax), %edi
	vfmadd231ss	16(%r12,%r15), %xmm7, %xmm0
	cmpl	%edi, %ebx
	jle	.L57
	addl	%edx, %edi
	addl	$6, %eax
	movslq	%edi, %rdi
	vmovss	0(%r13,%rdi,4), %xmm5
	vfmadd231ss	20(%r12,%r15), %xmm5, %xmm0
	cmpl	%eax, %ebx
	jle	.L57
	addl	%edx, %eax
	cltq
	vmovss	0(%r13,%rax,4), %xmm7
	vfmadd231ss	24(%r12,%r15), %xmm7, %xmm0
.L57:
	vmovss	%xmm0, (%r8)
	addq	$4, %r8
	addl	%ecx, %edx
	cmpq	%r9, %r8
	jne	.L49
	xorl	%eax, %eax
	vzeroupper
	call	get_time_millis
	vsubsd	-96(%rbp), %xmm0, %xmm4
	xorl	%eax, %eax
	vmovq	%xmm4, %rbx
	call	get_time_millis
	movl	-100(%rbp), %ecx
	movl	-64(%rbp), %r8d
	movq	%r12, %rsi
	movq	-80(%rbp), %rdx
	movq	%r13, %rdi
	vmovsd	%xmm0, -56(%rbp)
	call	matvec_avx512
	xorl	%eax, %eax
	call	get_time_millis
	vsubsd	-56(%rbp), %xmm0, %xmm4
	movq	-88(%rbp), %rcx
	xorl	%eax, %eax
	movq	-80(%rbp), %rsi
	movq	-72(%rbp), %rdi
	vmovss	.LC5(%rip), %xmm5
	vmovsd	%xmm4, -56(%rbp)
	vxorps	%xmm4, %xmm4, %xmm4
	vmovaps	%xmm4, %xmm1
	jmp	.L62
	.p2align 4,,10
	.p2align 3
.L65:
	movq	%rdx, %rax
.L62:
	vmovss	(%rcx,%rax,4), %xmm0
	vsubss	(%rsi,%rax,4), %xmm0, %xmm0
	leaq	1(%rax), %rdx
	vmovaps	%xmm0, %xmm2
	vxorps	%xmm5, %xmm0, %xmm3
	vcmpltss	%xmm4, %xmm0, %xmm0
	vblendvps	%xmm0, %xmm3, %xmm2, %xmm0
	vmaxss	%xmm1, %xmm0, %xmm1
	cmpq	%rdi, %rax
	jne	.L65
.L63:
	vmovq	%rbx, %xmm0
	movl	$.LC6, %edi
	movl	$1, %eax
	vmovss	%xmm1, -64(%rbp)
	call	printf
	movl	$.LC7, %edi
	movl	$1, %eax
	vmovsd	-56(%rbp), %xmm0
	call	printf
	movl	$.LC8, %edi
	movl	$1, %eax
	vmovq	%rbx, %xmm5
	vdivsd	-56(%rbp), %xmm5, %xmm0
	call	printf
	movl	$.LC9, %edi
	vmovss	-64(%rbp), %xmm1
	movl	$1, %eax
	vcvtss2sd	%xmm1, %xmm1, %xmm0
	call	printf
	xorl	%eax, %eax
.L41:
	addq	$64, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L64:
	.cfi_restore_state
	xorl	%edi, %edi
	vmovaps	%xmm4, %xmm0
	xorl	%eax, %eax
	jmp	.L50
.L72:
	movq	(%rsi), %rsi
	movl	$.LC3, %edi
	xorl	%eax, %eax
	call	printf
	movl	$1, %eax
	jmp	.L41
.L48:
	xorl	%eax, %eax
	call	get_time_millis
	vsubsd	-96(%rbp), %xmm0, %xmm4
	xorl	%eax, %eax
	vmovq	%xmm4, %rbx
	call	get_time_millis
	movl	-64(%rbp), %r8d
	movl	-100(%rbp), %ecx
	movq	%r12, %rsi
	movq	-80(%rbp), %rdx
	movq	%r13, %rdi
	vmovsd	%xmm0, -56(%rbp)
	call	matvec_avx512
	xorl	%eax, %eax
	call	get_time_millis
	vsubsd	-56(%rbp), %xmm0, %xmm4
	vxorps	%xmm1, %xmm1, %xmm1
	vmovsd	%xmm4, -56(%rbp)
	jmp	.L63
	.cfi_endproc
.LFE5498:
	.size	main, .-main
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	1093567616
	.align 8
.LC1:
	.long	0
	.long	1083129856
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC4:
	.long	1120403456
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC5:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.ident	"GCC: (GNU) 11.4.1 20230605 (Red Hat 11.4.1-2)"
	.section	.note.GNU-stack,"",@progbits
