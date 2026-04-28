using Revise, Novikov, FFTW, Plots

function integrate(u, L, N)
    dx = L / N
    # must count first element twice since right endpoint is omitted for Fourier
    sum = 0.5 * dx * (u[1] + u[2])
    for i = 1:N-1
        sum += 0.5 * dx * (u[i] + u[i+1])
    end
    return sum
end

N = 1024
L = 2 * pi
t_f = 1.0
q = convert(Int, 10000 * t_f)
kvec = gen_kvec(L, N)

u(x) = cos(x)
x, u_0 = dscrt(u, L, N)
au_f = evolve(u_0, t_f, q, kvec, N)
plot(x, u_0)
display(plot!(x, au_f))

mass_1 = 1 / L * integrate(u_0, L, N)
mass_2 = 1 / L * integrate(au_f, L, N)

mom_1 = 1 / L * integrate(u_0 .^ 2, L, N)
mom_2 = 1 / L * integrate(au_f .^ 2, L, N)

energy_1 = integrate(u_0 .^ 2, L, N)
energy = integrate(au_f .^ 2, L, N)
println("Conserved quantities - values show difference from IC to computed
        solution at t = ", t_f)
println("CQ1 - Mass:\t", abs(mass_2 - mass_1))
println("CQ2 - Momentum:\t", abs(mom_2 - mom_1))



