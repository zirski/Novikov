using FFTW, LinearAlgebra, Dates
"""
    kvec(L, N)

Return the spectral differentiation multipliers `2πik/L` for `k = 0:div(N, 2)`,
matching the output of `rfft` on an `N`-point grid over a period of length `L`.
"""
kvec(L::Float64, N::Int64) = [(im * 2 * pi * k) / L for k = 0:div(N, 2)]

"""
    deriv!(u, du, p, uhat, kvec, plan, iplan)

Compute the `p`th derivative of the periodic function `u` spectrally and store
it in `du`; `u` is not mutated.

# Arguments
- `u`: function values to be differentiated
- `du`: output buffer for the `p`th derivative of `u`
- `p`: order of the derivative (must be non-negative)
- `uhat`: complex scratch buffer of the same length as `kvec`
- `kvec`: differentiation multipliers, see [`kvec`](@ref)
- `plan`, `iplan`: forward (`rfft`) and inverse (`irfft`) FFT plans
"""
@inline function deriv!(
    u::AbstractArray{T},
    du::AbstractArray{T},
    p::Integer,
    uhat::AbstractArray{S},
    kvec::AbstractArray{S},
    plan,
    iplan,
) where {T<:Real,S<:Complex}
    p < zero(p) && throw(ArgumentError("Exponent p must be positive."))
    length(uhat) != length(kvec) &&
        throw(ArgumentError("uhat and kvec vectors must be of equivalent size."))
    length(u) != length(du) && throw(
        ArgumentError(
            "u and du vectors must be of equivalent size; u has size " *
            string(length(u)) *
            " and du has size " *
            string(length(du)) *
            ".",
        ),
    )

    mul!(uhat, plan, u)
    @. uhat = uhat * kvec^p
    mul!(du, iplan, uhat)

    return nothing
end

"""
    deriv(u, uhat, p, kvec, plan, iplan)

Allocating version of [`deriv!`](@ref): return the `p`th derivative of `u` as a
new vector.
"""
@inline function deriv(
    u::AbstractArray{T},
    uhat::AbstractArray{S},
    p::Int,
    kvec::AbstractArray{S},
    plan,
    iplan,
) where {T<:Real,S<:Complex}
    du = Vector{S}(undef, length(u))
    deriv!(u, du, p, uhat, kvec, plan, iplan)

    return du
end

"""
    rk4!(f!, uhat, u, uhat_tmp, dus, t, q, ks, plan, iplan)

Integrate the autonomous system `uhat' = f(uhat)` from time 0 to `t` with `q`
steps of classical fourth-order Runge-Kutta. `uhat` is updated in place and no
allocations are made.

# Arguments
- `f!`: rate function, called as `f!(u, k, u_x, u_xx, u_xxx, plan, iplan)`
- `uhat`: complex (Fourier-space) state to be integrated, mutated in place
- `u`: scratch buffer for the real-valued (physical-space) state
- `uhat_tmp`: generic complex scratch buffer
- `dus`: `N × 3` scratch buffer for the derivatives of `u`
- `t`: final time
- `q`: number of steps (not named `n` to avoid confusion with the array size `N`)
- `ks`: `length(uhat) × 4` scratch buffer for the Runge-Kutta stages
- `plan`, `iplan`: forward and inverse FFT plans
"""
function rk4!(
    f!::Function,
    uhat::AbstractArray{S},
    u::AbstractArray{T},
    uhat_tmp::AbstractArray{S},
    dus::AbstractArray{T},
    t,
    q::Int,
    ks,
    plan,
    iplan,
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
        @views @. uhat =
            uhat + (dt / 6) * (ks[:, 1] + 2 * (ks[:, 2] + ks[:, 3]) + ks[:, 4])
    end
    return nothing
end

"""
    dscrt(f, a, L, N)

Sample `f` at `N` equally spaced points on `[a, a + L)`. Return a named tuple
`(x, y)` of the sample points and the values of `f` at them.
"""
function dscrt(f, a, L, N)
    xvec = collect(0:(N-1)) * (L / N) .+ a
    return (x = xvec, y = f.(xvec))
end

"""
    dscrt(f, L, N)

Sample `f` at `N` equally spaced points on `[0, L)`. Return a named tuple
`(x, y)` of the sample points and the values of `f` at them.
"""
function dscrt(f, L, N)
    xvec = collect(0:(N-1)) * (L / N)
    return (x = xvec, y = f.(xvec))
end

"""
    integrate(u, L, N)

Integrate the periodic function sampled as `u` (`N` points over a period of
length `L`, right endpoint omitted) with the rectangle rule.
"""
function integrate(u, L, N)
    dx = L / N
    # must count first element twice since right endpoint is omitted for Fourier
    sum = u[1] * dx
    for i = 2:N
        sum += dx * u[i]
    end
    return sum
end

"""
    log(msg)

Append `msg` to the log file at `LOG_PATH`, prefixed with the current time.
"""
function log(msg)
    open(LOG_PATH, "a") do io
        write(io, string(now()) * ": " * msg * "\n")
    end
end

"""
    istw(sol, c, L)

Check whether `sol` is a traveling wave with speed `c` and period `L` by
evolving it for one period of travel (time `L / c`) and returning the 2-norm of
the difference between the initial and final states. A result near zero
indicates a traveling wave.
"""
function istw(sol::Vector{Float64}, c::Float64, L::Float64)
    au_f = evolve(sol, L / c, round(1000 * L / c), kvec(L, length(sol)))
    return norm(abs.(sol .- au_f))
end

"""
    istw(sol::Union{NovikovProblem,NovikovSolution})

Call `istw(sol.sol, sol.c, sol.L)`.
"""
function istw(sol::T) where {T<:Union{NovikovProblem,NovikovSolution}}
    return istw(sol.sol, sol.c, sol.L)
end
