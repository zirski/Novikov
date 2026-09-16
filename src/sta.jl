using LinearAlgebra
using Base.Threads
using FFTW

@inline get_uhat(uhat, i, mean) = i == 0 ? mean : i < 0 ? uhat[-i] : uhat[i]

@inline vcoef(k, mu, L) = 2pi * k / L + mu
@inline ucoef(k, L) = 2pi * k / L

@inline t1(k, uhat, mean, c, L, mu) = c * (vcoef(k, mu, L) + vcoef(k, mu, L)^3)

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

function l_element(k, n, N, args...)
    if k == n
        return t1(k, args...) - sum(j -> t2(k, j, n, args...), -N+k:N+k)
    else
        return sum(max(-2N, -N + k):min(2N, N + k)) do j
            max(-N, -N + j) <= n <= min(N, N + j) && return t2(k, j, n, args...)
            return 0
        end
    end
end

function fill_lmat!(lmat, N, args...)
    @threads for n = -N:N
        for k = -N:N
            lmat[k+N+1, n+N+1] = l_element(k, n, N, args...)
        end
    end
end

"""
    compute_evals(N, uhat, mean, c, L, mu)

Never use this method. 
"""
function compute_evals(N, uhat, mean, c, L, mu; evecs = false)
    args = (uhat, mean, c, L, mu)

    M = Diagonal([-im * (1 + vcoef(m, mu, L)^2) for m = -N:N])
    lmat = Array{ComplexF64,2}(undef, 2N + 1, 2N + 1)
    fill_lmat!(lmat, N, args...)

    probmat = inv(M) * lmat

    evecs && return eigen(probmat)
    return eigvals(probmat)
end

"""
    compute_evals(sol, mu; minmodes=128, mtol=1e-12)

Compute the eigenvalues of a perturbed solution `sol` to the NE with the Floquet
parameter `mu`.
"""
function compute_evals(
    sol::NovikovSolution,
    mu;
    minmodes = 128,
    mtol = 1e-12,
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

