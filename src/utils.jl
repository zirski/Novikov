using FFTW, LinearAlgebra

# Generates vector of complex values to be applied during derivative calculations. 
gen_kvec(L::Float64, N::Int64) = [(im * 2 * pi * k) / L for k = 0:div(N, 2)]

# Computes pth derivative of a function using fft technique and stores result in 
# du; does not mutate u.
# u:    function vector to be differentiated
# du:   stores pth derivative of u
# uhat: scratch buffer
function deriv!(
    u::AbstractArray{Float64,1},
    du::AbstractArray{Float64,1},
    p::Int,
    uhat::AbstractArray{ComplexF64,1},
    kvec::AbstractArray{ComplexF64,1},
    plan,
    iplan
)
    mul!(uhat, plan, u)
    @. uhat = uhat * kvec^p
    mul!(du, iplan, uhat)
    return nothing
end

# 1st-derivative version; faster for single derivatives 
function deriv!(
    u::AbstractArray{Float64,1},
    du::AbstractArray{Float64,1},
    uhat::AbstractArray{ComplexF64,1},
    kvec::AbstractArray{ComplexF64,1},
    plan,
    iplan
)
    mul!(uhat, plan, u)
    @. uhat = uhat * kvec
    mul!(du, iplan, uhat)
    return nothing
end

function deriv(
    u::AbstractArray{Float64,1},
    uhat::AbstractArray{ComplexF64,1},
    kvec::AbstractArray{ComplexF64,1},
    plan,
    iplan
)
    mul!(uhat, plan, u)
    @. uhat = uhat * kvec
    return iplan * uhat
end

function deriv(
    u::AbstractArray{Float64,1},
    uhat::AbstractArray{ComplexF64,1},
    p::Int,
    kvec::AbstractArray{ComplexF64,1},
    plan,
    iplan
)
    mul!(uhat, plan, u)
    @. uhat = uhat * kvec^p
    return iplan * uhat
end

# Vector-valued rk4 (autonomous)
# Mutates input vector u in-place; 0 allocations
# f:        Vector-valued vectorized function
# uhat:     Complex input vector to be integrated
# u_func:   scratch buffer for real-valued u
# u_tmp:    generic complex scratch buffer
# dus:      scratch buffer for u derivs
# ks:       2d array of scratch buffers
# q:        number of iterations. Not named n to avoid confusion with N (global 
#           array size).
function rk4!(
    f!::Function,
    uhat::Vector{ComplexF64},
    u::Vector{Float64},
    u_tmp::Vector{ComplexF64},
    dus::Array{Float64,2},
    t,
    q,
    ks,
    plan,
    iplan
)
    dt = t / q
    dtd2 = 0.5 * dt
    # serves two purposes: stores each k after each f! call, and stores uhat in
    # between
    for _ = 1:q
        mul!(u, iplan, uhat)
        # f! preserves the state of uhat while computing the derivative to be 
        # stored in ks. Necessary because we need an untainted uhat for line 87.
        # This also necessitates u_tmp.
        @views f!(u, ks[:, 1], dus[:, 1], dus[:, 2], dus[:, 3], plan, iplan)

        @views @. u_tmp = dtd2 * ks[:, 1] + uhat
        mul!(u, iplan, u_tmp)
        @views f!(u, ks[:, 2], dus[:, 1], dus[:, 2], dus[:, 3], plan, iplan)

        @views @. u_tmp = dtd2 * ks[:, 2] + uhat
        mul!(u, iplan, u_tmp)
        @views f!(u, ks[:, 3], dus[:, 1], dus[:, 2], dus[:, 3], plan, iplan)

        @views @. u_tmp = dt * ks[:, 3] + uhat
        mul!(u, iplan, u_tmp)
        @views f!(u, ks[:, 4], dus[:, 1], dus[:, 2], dus[:, 3], plan, iplan)

        # update step.
        @views @. uhat = uhat + (dt / 6) * (ks[:, 1] + 2 * (ks[:, 2] + ks[:, 3])
                                            + ks[:, 4])
    end
    return nothing
end

function dscrt(f, a, L, N)
    xvec = collect(0:N-1) * (L / N) .+ a
    return (x=xvec, y=f.(xvec))
end

function dscrt(f, L, N)
    xvec = collect(0:N-1) * (L / N)
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