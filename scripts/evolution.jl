using Revise, Novikov, FFTW, Plots


N = 128
L = 2 * pi
t_f = 4.0
q = convert(Int, 1000 * t_f)
kvec = gen_kvec(L, N)

u(x, t) = cos(x)
x, u_0 = dscrt(x -> u(x, 0), L, N)
println(u_0[1:5])
au_f = evolve(u_0, t_f, q, kvec, N)
println(u_0[1:5])
display(plot(x, au_f))

# display(plot(x, abs.(au_f - u_f)))