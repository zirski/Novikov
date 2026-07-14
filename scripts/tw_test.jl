using Novikov, Revise, LinearAlgebra, FFTW, Plots, BenchmarkTools, LaTeXStrings

# TBD
c = 0.2689
N = 64
N_f = div(N, 2)
L = 2pi
q = 5
kvec = gen_kvec(L, N)

# your average cosine wave
fhat_guess = zeros(ComplexF64, N_f + 1)
# 1st fourier mode; 0th mode is left as 0 to enforce zero-mean solution
fhat_guess[2] = Complex(0.015, 0)
f_guess = irfft(fhat_guess, N)
xvec = collect(0:(N-1)) * (L / N) .- L / 2

fhat_newton_input = copy(fhat_guess[2:end])
fhat_coefs = zeros(ComplexF64, N_f + 1)
# only overwrite the 1st, 2nd, ..., N_fth modes
fhat_coefs[2:end] = gen_tw_sol_3(fhat_newton_input, c, L, N_f, q)
# @profview gen_tw_sol_2(fhat_guess, c, L, N_f, q)
# @btime gen_tw_sol_3(fhat_guess, c, L, N_f, q)

# display(plot(xvec[1:N_f], real(fhat_guess), label="Re(guess)"))
# display(plot!(xvec[1:N_f], imag(fhat_guess), label="Im(guess)"))

# fhat_coefs ./= N

# display(plot(xvec[1:N_f], real(fhat_coefs), label="Re(fhat)"))
# display(plot!(xvec[1:N_f], imag(fhat_coefs), label="Im(fhat)"))


ic = irfft(fhat_coefs, N)

display(plot(xvec, f_guess, label="initial guess"))
display(plot(xvec, ic, label="ic", xlabel=L"\xi", ylabel=L"f(\xi)"))

# savefig("~/research/presentations/26.06.02_novikov_tw/badgraph.pdf")

# au_f = evolve(ic, 100.0, 10000, kvec, N)

# println(norm(abs.(ic .- au_f)))

# display(plot(xvec, au_f, label="auf at t=100"))