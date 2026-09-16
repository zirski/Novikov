using FFTW, LinearAlgebra

function evolve(
    u::AbstractVector{R},
    t_f::Real,
    q::Real,
    kvec::AbstractVector{C},
) where {R<:Real, C<:Complex}
    t_f < 0 && throw(ArgumentError("Final time cannot be negative."))

    N = length(u)
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
        u::AbstractVector{R},
        u_output::AbstractVector{C},
        u_x::AbstractVector{R},
        u_xx::AbstractVector{R},
        u_xxx::AbstractVector{R},
        plan,
        iplan
    ) where {R<:Real, C<:Complex}
        deriv!(u, u_x, 1, uhat_buf, kvec, plan, iplan)
        deriv!(u, u_xx, 2, uhat_buf, kvec, plan, iplan)
        deriv!(u, u_xxx, 3, uhat_buf, kvec, plan, iplan)

        @. u = -u ^ 2 * (4 * u_x - u_xxx) + 3 * u * u_x * u_xx
        mul!(u_output, plan, u)
        @. u_output /= (1 - kvsquared)
        return nothing
    end

    try
        rk4!(f!, uhat_out, u_func, uhat_tmp, dus, t_f, Int(q), ks, plan, iplan)
    catch e
        if e isa ArgumentError
            throw(ArgumentError("q must be convertable to an integral type."))
        else
            error(e)
        end
    end
    mul!(u_func, iplan, uhat_out)
    return u_func
end
