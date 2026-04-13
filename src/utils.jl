using FFTW, LinearAlgebra

# Computes 1st derivative of a function using fft technique and stores result in du; does not mutate u.
function deriv!(u, du, p, uhat, kvec, plan, iplan)
    mul!(uhat, plan, u)
    @. uhat = uhat * kvec^p
    mul!(du, iplan, uhat)
    return nothing
end

# Vector-valued rk4 (autonomous)
# Updates input vector u in-place; 0 allocations
# f: Vector-valued vectorized function
# ks: 2d array of scratch buffers
# q: Number of iterations. Not named n to avoid confusing with N (global array size).
function rk4!(f!::Function, u, u_tmp, dus, t, q, ks)
    dt = t / q
    dtd2 = 0.5 * dt
    for i = 1:q
        # @views is Julia's cumbersome way to avoid making unnecessary allocations. It essentially forces a read of any data on
        # the right-hand side of the equals sign; normally taking any slice of an array allocates a buffer before copying data
        # over to the left-hand side.
        # @. applies broadcasting to all operators after it. Julia's vectorization is nice and will optimize certain loop
        # operations, not sure how.
        #
        # f! preserves the state of u while computing the derivative to be stored in ks. Necessary because we need an untainted
        # u for line 34. 
        @views f!(ks[:, 1], u, dus[1], dus[2], dus[3])
        @views @. u_tmp = u + dtd2 * ks[:, 1]
        @views f!(ks[:, 2], u_tmp, dus[1], dus[2], dus[3])
        @views @. u_tmp = dtd2 * ks[:, 2] + u
        @views f!(ks[:, 3], u_tmp, dus[1], dus[2], dus[3])
        @views @. u_tmp = dt * ks[:, 3] + u
        @views f!(ks[:, 4], u_tmp, dus[1], dus[2], dus[3])

        # update step. We still need @views here since we're taking slices of ks.
        @views @. u = u + (dt / 6) * (ks[:, 1] + 2 * (ks[:, 2] + ks[:, 3]) + ks[:, 4])
    end
    return nothing
end