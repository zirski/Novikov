module Novikov

include("utils.jl")
include("tw.jl")

export gen_kvec, evolve, dscrt, integrate, deriv!, deriv, gen_tw_sol_2,
    gen_tw_sol_1, gen_tw_sol_3, gen_jacobian_1, gen_jacobian_2, gen_f_from_fhats, f_fourier, F, print_jac

using FFTW, LinearAlgebra

function evolve(
    u::Vector{Float64},
    t_f::Float64,
    q::Int64,
    kvec::Vector{ComplexF64},
    N::Int64
)
    Ndiv2 = div(N, 2)
    uhat_buf = Vector{ComplexF64}(undef, Ndiv2 + 1)
    u_func = similar(u)
    plan = plan_rfft(u)
    iplan = plan_irfft(uhat_buf, N)
    uhat_out = plan * u

    # rk4 preallocations
    ks = zeros(ComplexF64, Ndiv2 + 1, 4)
    uhat_tmp = similar(uhat_buf)
    kvsquared = kvec .^ 2

    # scratch buffer for u derivatives: [u_x, u_xx, u_xxx]
    dus = zeros(Float64, N, 3)

    # rate of change function (1/(1+k^2) * ghat)
    # We need to perform the derivatives in function space to compute g
    # accurately, which sucks for time efficiency but here we are
    function f!(
        u::AbstractArray{Float64,1},
        u_output::AbstractArray{ComplexF64,1},
        u_x::AbstractArray{Float64,1},
        u_xx::AbstractArray{Float64,1},
        u_xxx::AbstractArray{Float64,1},
        plan,
        iplan
    )
        deriv!(u, u_x, uhat_buf, kvec, plan, iplan)
        deriv!(u, u_xx, 2, uhat_buf, kvec, plan, iplan)
        deriv!(u, u_xxx, 3, uhat_buf, kvec, plan, iplan)

        @. u = -u ^ 2 * (4 * u_x - u_xxx) + 3 * u * u_x * u_xx
        mul!(u_output, plan, u)
        @. u_output /= (1 - kvsquared)
        return nothing
    end

    rk4!(f!, uhat_out, u_func, uhat_tmp, dus, t_f, q, ks, plan, iplan)
    mul!(u_func, iplan, uhat_out)
    return u_func
end

end # module Novikov
