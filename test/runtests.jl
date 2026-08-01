using Test
using Novikov
using LinearAlgebra

@testset "tw" begin
    N_f = 64
    N = 4096
    L = 2pi
    c = 0.2689
    fhat_guess = zeros(ComplexF64, N_f)
    fhat_guess[1] = 0.3
    fhat_guess[2] = 0.05

    prob = construct_twsol(fhat_guess, c, L, N_gp=N)
    au_f = evolve(prob.sol, L / c, 4000, gen_kvec(L, N), N)
    diff = norm(abs.(prob.sol .- au_f))
    @test isapprox(diff, 0, atol=1e-10)
end
