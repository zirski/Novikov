using LinearAlgebra
using Base.Threads
using FFTW

# U is real, so Û(-i) = conj(Û(i))
@inline get_uhat(uhat, i, mean) = i == 0 ? mean : i < 0 ? conj(uhat[-i]) : uhat[i]

@inline vcoef(k, mu, L) = 2pi * k / L + mu
@inline ucoef(k, L) = 2pi * k / L

@inline t1(k, uhat, mean, c, L, mu) = c * (vcoef(k, mu, L) + vcoef(k, mu, L)^3)

# Every T2 sum over j is a convolution in a = k - j, b = j - l with a + b = k - l,
# so the sums depend only on p = k - l (see notes/latex/fast_assembly.tex).
# Returns a (4N + 1) x 5 array whose row p + 2N + 1 holds S0, S1, S2, S3, S12 at p.
function conv_sums(N, uhat, mean, L)
    S = zeros(ComplexF64, 4N + 1, 5)
    @threads for p = (-2N):(2N)
        # both Û indices stay in -N:N by construction
        for a = max(-N, p - N):min(N, p + N)
            b = p - a
            w = get_uhat(uhat, a, mean) * get_uhat(uhat, b, mean)
            qa, qb = ucoef(a, L), ucoef(b, L)
            S[p+2N+1, 1] += w
            S[p+2N+1, 2] += w * qb
            S[p+2N+1, 3] += w * qb^2
            S[p+2N+1, 4] += w * qb^3
            S[p+2N+1, 5] += w * qa * qb^2
        end
    end
    return S
end

function fill_lmat!(lmat, N, uhat, mean, c, L, mu)
    S = conv_sums(N, uhat, mean, L)
    @threads for n = (-N):N
        kn = vcoef(n, mu, L)
        for k = (-N):N
            p = k - n + 2N + 1
            lmat[k+N+1, n+N+1] =
                (k == n ? t1(n, uhat, mean, c, L, mu) : zero(ComplexF64)) - (
                    (4kn + kn^3) * S[p, 1] +
                    (8 + 3kn^2) * S[p, 2] +
                    3kn * S[p, 3] +
                    2S[p, 4] +
                    3S[p, 5]
                )
        end
    end
end

"""
    compute_evals(N, uhat, mean, c, L, mu)

Never use this method. 
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
    compute_evals(sol, mu; minmodes=128, mtol=1e-12, evecs=false)

Compute the eigenvalues of a perturbed solution `sol` to the NE with the Floquet
parameter `mu`. Fourier modes of `sol` are halved away while every discarded
coefficient is below `mtol`, keeping at least `minmodes` modes. With
`evecs=true`, return the `Eigen` factorization instead of only the eigenvalues.
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

    # only halve if every mode being discarded is negligible, and never go below
    # minmodes
    while true
        newmax = div(maxmode, 2) + 1
        (newmax - 1 < minmodes || maximum(abs, uhat[(newmax+1):maxmode]) >= mtol) &&
            break
        maxmode = newmax
    end

    # never use the Nyquist coefficient; it would be counted as both Û(N) and Û(-N)
    maxmode == length(uhat) && (maxmode -= 1)

    log(
        "initiating eval computation with " *
            Base.string(maxmode) *
            " unique modes",
    )
    return compute_evals(
        maxmode - 1,
        uhat[2:maxmode],
        mean,
        sol.c,
        sol.L,
        mu;
        kwargs...,
    )
end

