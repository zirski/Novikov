module Utils

using FFTW, LinearAlgebra
# Generates vector of complex values to be applied during derivative calculations. 
gen_kvec(L::Float64, N::Int64) = [(im * 2 * pi * k) / L for k = 0:div(N, 2)]

# Computes pth derivative of a function using fft technique and stores result in 
# du; does not mutate u.
# u:    function vector to be differentiated
# du:   stores pth derivative of u
# uhat: scratch buffer
@inline function deriv!(
    u::AbstractArray{T},
    du::AbstractArray{T},
    p::Integer,
    uhat::AbstractArray{S},
    kvec::AbstractArray{S},
    plan,
    iplan
) where {T<:Real,S<:Complex}
    p < zero(p) && throw(DomainError(p, "Exponent p must be positive."))
    length(uhat) != length(kvec) && throw(ArgumentError("uhat and kvec vectors must be of equivalent size."))
    length(u) != length(du) && throw(ArgumentError("u and du vectors must be of equivalent size; u has size " * string(length(u)) * " and du has size " * string(length(du)) * "."))

    mul!(uhat, plan, u)
    @. uhat = uhat * kvec^p
    mul!(du, iplan, uhat)

    return nothing
end

@inline function deriv(
    u::AbstractArray{T},
    uhat::AbstractArray{S},
    p::Integer,
    kvec::AbstractArray{S},
    plan,
    iplan
) where {T<:Real,S<:Complex}

    du = Vector{S}(undef, length(u))
    deriv!(u, du, p, uhat, kvec, plan, iplan)

    return du
end

# Vector-valued rk4 (autonomous)
# Mutates input vector u in-place; 0 allocations
# f:        Vector-valued vectorized function
# uhat:     Complex input vector to be integrated
# u:        scratch buffer for real-valued u
# uhat_tmp: generic complex scratch buffer
# dus:      scratch buffer for u derivs
# ks:       2d array of scratch buffers
# q:        number of iterations. Not named n to avoid confusion with N (global 
#           array size).
function rk4!(
    f!::Function,
    uhat::AbstractArray{S},
    u::AbstractArray{T},
    uhat_tmp::AbstractArray{S},
    dus::AbstractArray{T},
    t,
    q,
    ks,
    plan,
    iplan
) where {T<:Real,S<:Complex}
    dt = t / q
    dtd2 = 0.5 * dt
    # serves two purposes: stores each k after each f! call, and stores uhat in
    # between
    for _ = 1:q
        mul!(u, iplan, uhat)
        # f! preserves the state of uhat while computing the derivative to be 
        # stored in ks. Necessary because we need an untainted uhat for line 112.
        # This also necessitates uhat_tmp.
        @views f!(u, ks[:, 1], dus[:, 1], dus[:, 2], dus[:, 3], plan, iplan)

        @views @. uhat_tmp = dtd2 * ks[:, 1] + uhat
        mul!(u, iplan, uhat_tmp)
        @views f!(u, ks[:, 2], dus[:, 1], dus[:, 2], dus[:, 3], plan, iplan)

        @views @. uhat_tmp = dtd2 * ks[:, 2] + uhat
        mul!(u, iplan, uhat_tmp)
        @views f!(u, ks[:, 3], dus[:, 1], dus[:, 2], dus[:, 3], plan, iplan)

        @views @. uhat_tmp = dt * ks[:, 3] + uhat
        mul!(u, iplan, uhat_tmp)
        @views f!(u, ks[:, 4], dus[:, 1], dus[:, 2], dus[:, 3], plan, iplan)

        # update step.
        @views @. uhat = uhat + (dt / 6) * (ks[:, 1] + 2 * (ks[:, 2] + ks[:, 3])
                                            + ks[:, 4])
    end
    return nothing
end

function dscrt(f, a, L, N)
    xvec = collect(0:(N-1)) * (L / N) .+ a
    return (x=xvec, y=f.(xvec))
end

function dscrt(f, L, N)
    xvec = collect(0:(N-1)) * (L / N)
    return (x=xvec, y=f.(xvec))
end

function integrate(u, L, N)
    dx = L / N
    # must count first element twice since right endpoint is omitted for Fourier
    sum = u[1] * dx
    for i = 2:N
        sum += dx * u[i]
    end
    return sum
end

end
