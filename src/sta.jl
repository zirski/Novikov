using LinearAlgebra
using Base.Threads
using FFTW

@inline get_uhat(arr, i; mean=0.3) = i == 0 ? mean : i < 0 ? arr[-i] : arr[i]

@inline t1(m, uhat, c, L, mu) = c * (vcoef(m, mu, L) + vcoef(m, mu, L)^3)

@inline vcoef(k, mu, L) = 2pi * k / L + mu
@inline ucoef(k, L) = 2pi * k / L

@inline function t2(m, j, n, uhat, c, L, mu)
    return (
        4vcoef(n, mu, L) + vcoef(n, mu, L)^3 + 8ucoef(j - n, L) + 2ucoef(m - j, L)^3
        -
        3(ucoef(m - j, L) * ucoef(j - n, L)^2 + ucoef(j - n, L)^2 * vcoef(n, mu, L) + ucoef(j - n, L) * vcoef(n, mu, L)^2)
    )
    *
    get_uhat(uhat, m - j) * get_uhat(uhat, j - n)
end

function l_element(m, n, N, args...)
    if m == n
        return t1(m, args...) - sum(j -> t2(m, j, n, args...), -N:N)
    else
        return -sum(j -> t2(m, j, n, args...), max(-2N, -N + m):min(2N, N + m))
    end
end

function fill_lmat!(lmat, N, args...)
    @threads for n in 1:2N + 1
        for m in 1:2N + 1
            lmat[m, n] = l_element(m, n, N, args...)
        end
    end
end

function compute_evals(N, uhat, c, L, mu)
    args = (uhat, c, L, mu)

    M_inv = Diagonal([(-im * (1 + vcoef(m, mu, L)^2))^ -1 for m in -N:N])
    lmat = Array{ComplexF64,2}(undef, 2N + 1, 2N + 1)
    fill_lmat!(lmat, N, args...)

    probmat = M_inv * lmat
    return eigvals(probmat)
end

function compute_evals(sol::NovikovSolution, mu)
    #DEBUG
    nmodes = 64
    uhat = rfft(sol.sol)[1:nmodes] / sol.N
    return compute_evals(nmodes, uhat, sol.c, sol.L, mu)
end

