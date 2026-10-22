using LinearAlgebra
using Base.Threads
using FFTW

"""
    get_uhat(uhat, i, mean)

Return the Fourier coefficient of index `i`: `mean` for `i == 0`, otherwise
`uhat[abs(i)]`.
"""
@inline get_uhat(uhat, i, mean) = i == 0 ? mean : i < 0 ? uhat[-i] : uhat[i]

"""
    vcoef(k, mu, L)

Wavenumber `2πk/L + mu` of the `k`th mode of a perturbation with Floquet
parameter `mu`.
"""
@inline vcoef(k, mu, L) = 2pi * k / L + mu
"""
    ucoef(k, L)

Wavenumber `2πk/L` of the `k`th mode of the base solution.
"""
@inline ucoef(k, L) = 2pi * k / L

"""
    t1(k, uhat, mean, c, L, mu)

Linear (wave-speed) contribution to the diagonal element `k` of the linearized
operator.
"""
@inline t1(k, uhat, mean, c, L, mu) = c * (vcoef(k, mu, L) + vcoef(k, mu, L)^3)

"""
    t2(k, j, l, uhat, mean, c, L, mu)

Term `j` of the convolution sum contributing to element `(k, l)` of the
linearized operator.
"""
@inline function t2(k, j, l, uhat, mean, c, L, mu)
    return (*(
        +(
            4vcoef(l, mu, L) +
                vcoef(l, mu, L)^3 +
                8ucoef(j - l, L) +
                2ucoef(k - j, L)^3,
            3(
                ucoef(k - j, L) * ucoef(j - l, L)^2 +
                    ucoef(j - l, L)^2 * vcoef(l, mu, L) +
                    ucoef(j - l, L) * vcoef(l, mu, L)^2
            ),
        ),
        get_uhat(uhat, k - j, mean) * get_uhat(uhat, j - l, mean),
    ))
end

"""
    l_element(k, n, N, uhat, mean, c, L, mu)

Compute element `(k, n)` of the linearized operator truncated to modes `-N:N`.
"""
function l_element(k, n, N, args...)
    if k == n
        return t1(k, args...) - sum(j -> t2(k, j, n, args...), (-N+k):(N+k))
    else
        return -sum((-N+k):(N+k)) do j
            abs(j - n) <= N && return t2(k, j, n, args...)
            return 0
        end
    end
end

"""
    fill_lmat!(lmat, N, uhat, mean, c, L, mu)

Fill the `(2N + 1) × (2N + 1)` matrix `lmat` with the linearized operator over
modes `-N:N` (multithreaded over columns).
"""
function fill_lmat!(lmat, N, args...)
    @threads for n = (-N):N
        for k = (-N):N
            lmat[k+N+1, n+N+1] = l_element(k, n, N, args...)
        end
    end
end

"""
    compute_evals(N, uhat, mean, c, L, mu; evecs=false)

Never use this method; call `compute_evals(sol, mu)` instead.

Compute the spectrum of `M⁻¹L`, the linearization about the solution with
Fourier coefficients `uhat`, mean `mean`, speed `c` and period `L`, truncated to
modes `-N:N`, for Floquet parameter `mu`. Return the eigenvalues, or an `Eigen`
factorization if `evecs` is `true`.
"""
function compute_evals(N, uhat, mean, c, L, mu; evecs=false)
    args = (uhat, mean, c, L, mu)

    M = Diagonal([-im * (1 + vcoef(m, mu, L)^2) for m = (-N):N])
    lmat = Array{ComplexF64,2}(undef, 2N + 1, 2N + 1)
    fill_lmat!(lmat, N, args...)

    probmat = inv(M) * lmat

    evecs && return eigen(probmat)
    return eigvals(probmat)
end

"""
    compute_evals(sol, mu; minmodes=128, mtol=1e-12, kwargs...)

Compute the eigenvalues of a perturbed solution `sol` to the NE with the Floquet
parameter `mu`.

The number of modes is halved (down to `minmodes`) while the highest modes sum
to less than `mtol`. Remaining keyword arguments (e.g. `evecs`) are passed to
the low-level method.
"""
function compute_evals(
    sol::NovikovSolution,
    mu;
    minmodes=128,
    mtol=1e-12,
    kwargs...,
)
    solvec = sol.sol

    mean = sum(solvec) / length(solvec)
    uhat = rfft(solvec) / length(solvec)

    maxmode = length(uhat)
    !isinteger(log2(maxmode - 1)) && throw(
        ArgumentError("input solution has an inappropriate number of modes"),
    )

    while maxmode > minmodes && abs(real(sum(uhat[maxmode-10:maxmode]))) < mtol
        maxmode = div(maxmode, 2) + 1
    end

    log(
        "initiating eval computation with " *
            Base.string(maxmode) *
            " unique modes",
    )
    return compute_evals(
        maxmode,
        uhat[2:maxmode],
        mean,
        sol.c,
        sol.L,
        mu;
        kwargs...,
    )
end

