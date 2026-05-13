# todo list:
# 1. make sure solver works (no mistakes)
# 2. test for conserved quantities (look at red notebook)
# 3. write code for traveling-wave IC generation (Novikov paper.pdf)
# 4. put random trig functions into Kdv to see what happens


module Novikov

include("utils.jl")

export gen_kvec, evolve, dscrt, integrate, deriv!, deriv

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
    u_tmp = similar(uhat_buf)
    kvsquared = kvec .^ 2

    # scratch buffer for u derivatives: [u_x, u_xx, u_xxx]
    dus = zeros(Float64, N, 3)

    # rate of change function (1/(1+k^2) * ghat)
    # We need to perform the derivatives in function space to compute g
    # accurately, which sucks for time efficiency but here we are
    function f!(
        u::Vector{Float64},
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

        u_output .= (plan * (-u .^ 2 .* (4 .* u_x - u_xxx) .+ 3 .* u .* u_x .* u_xx)) ./ (1 .- kvsquared)
        return nothing
    end

    rk4!(f!, uhat_out, u_func, u_tmp, dus, t_f, q, ks, plan, iplan)
    mul!(u_func, iplan, uhat_out)
    return u_func
end

end # module Novikov
