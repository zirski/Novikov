# todo list:
# 1. make sure solver works (no mistakes)
# 2. test for conserved quantities (look at red notebook)
# 3. write code for traveling-wave IC generation (Novikov paper.pdf)
# 4. put random trig functions into Kdv to see what happens


module Novikov

export gen_kvec, integrate

include("utils.jl")

using FFTW, LinearAlgebra

# Generates vector of complex values to be applied during derivative calculations. 
gen_kvec(L, N) = [(im * 2 * pi * k) / L for k = 0:div(N, 2)]

function integrate(u, t_f, q, kvec, N)
    uhat_buf = Vector{ComplexF64}(undef, Ndiv2 + 1)
    uhat = Vector{ComplexF64}(undef, Ndiv2 + 1)
    plan = plan_rfft(u)
    iplan = plan_irfft(uhat_buf, N)

    # rk4 preallocations
    ks = zeros(Float64, N, 4)
    u_tmp = similar(uhat_buf)

    # scratch buffer for u derivatives: [u_x, u_xx, u_xxx]
    dus = zeros(ComplexF64, N, 3)

    # rate of change function (1/(1+k^2) * g)
    function f!(u_output, u, u_x, u_xx, u_xxx)
        deriv!(u, u_x, 1, uhat_buf, kvec, plan, iplan)
        deriv!(u, u_xx, 2, uhat_buf, kvec, plan, iplan)
        deriv!(u, u_xxx, 3, uhat_buf, kvec, plan, iplan)

        #todo: make sure g(u) is in frequency space before multiplying by 1 / (1 + k^2)
        @. u_output = 1 / (1 + kvec^2) * (-u^2 * (4 * u_x - u_xxx) + 3 * u * u_x * u_xx)
    end
    rk4!(f!, uhat, u_tmp, dus, t_f, q, ks)
    mul!(u, iplan, uhat)
    return u
end

end # module Novikov
