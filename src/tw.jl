using LinearAlgebra
using Printf
using StyledStrings
using Base.Threads
using UUIDs

import Base: isless

"""
    alpha(c, n, lam)

Coefficient of the linear term of mode `n` in the traveling-wave equation with
speed `c` and base wavenumber `lam = 2π/L`.
"""
@inline alpha(c, n, lam) = -c * im * lam * n * (1 + (lam * n)^2)
"""
    beta(j, l, lam)

Coefficient of the cubic convolution term `fhat[k-j] * fhat[j-l] * fhat[l]` in
the traveling-wave equation.
"""
@inline beta(j, l, lam) =
    im * lam * l * (4 + (lam * l)^2 + 3 * lam^2 * l * (j - l))

"""
    gen_combo_indices(k, n)

Return the `(j, l, flag)` index triples of the convolution sum for mode `k` in
which two of the three `fhat` factors have index `±n`; combinations of three
are handled in the diagonal case of [`compute_elem`](@ref). `flag` indicates
which factor is left out (and included in the derivative term): 1 for vertical
(`l`), 2 for horizontal (`k - j`) and 3 for diagonal (`j - l`).
"""
@inline gen_combo_indices(k, n) = (
    # flag in last element corresponds to which fhat term is left out (and
    # included in the derivative term):
    # 
    # 1: vertical
    # 2: horizontal
    # 3: diagonal
    (k - n, n, 3),       # v1, h1
    (k + n, n, 3),       # v1, h2
    (2n, n, 2),          # v1, d1 
    (0, n, 2),           # v1, d2
    (k - n, k - 2n, 1),  # h1, d1 
    (k - n, k, 1),       # h1, d2
    (k + n, -n, 3),      # v2, h2
    (k - n, -n, 3),      # v2, h1
    (-2n, -n, 2),        # v2, d2
    (0, -n, 2),          # v2, d1
    (k + n, k + 2n, 1),  # h2, d2
    (k + n, k, 1),        # h2, d1
)

"""
    get_fhat(arr, i, mean)

Return the Fourier coefficient of index `i`: `mean` for `i == 0`, otherwise
`arr[abs(i)]`.
"""
@inline function get_fhat(arr, i, mean)
    i == 0 ? mean : i < 0 ? arr[-i] : arr[i]
end

"""
    check_combos(testcombo, combos)

Return whether the index pair `testcombo = (j, l)` matches the first two
entries of any element of `combos`.
"""
@inline function check_combos(testcombo, combos)
    for combo in combos
        testcombo == combo[1:2] && return true
    end
    return false
end

"""
    compute_elem(k, n, fhat, c, N, mean, lam)

Compute the Jacobian element `∂F_k/∂fhat_n` of the traveling-wave equation
(see [`F!`](@ref)) truncated to `N` modes.
"""
function compute_elem(k, n, fhat, c, N, mean, lam)
    term = zero(ComplexF64)
    combo_indices = gen_combo_indices(k, n)
    j_bounds = (-N + k, N + k)

    if k == n
        term =
            alpha(c, n, lam) +
            3 *
            (beta(0, n, lam) + beta(2n, n, lam) + beta(0, -n, lam)) *
            fhat[n]^2
    else
        # 2 - combinations
        for combo ∈ combo_indices
            # we're not counting the (2n, n) combo since it shares indices with
            # 3-combo and requires special logic
            if combo[1:2] != (2n, n)
                j = combo[1]
                l = combo[2]
                f = combo[3]
                if j_bounds[1] <= j <= j_bounds[2] &&
                   max(-N, -N + j) <= l <= min(N, N + j)
                    term_tmp = 2 * beta(j, l, lam) * fhat[n]
                    if f == 1
                        term_tmp *= get_fhat(fhat, l, mean)
                    elseif f == 2
                        term_tmp *= get_fhat(fhat, k - j, mean)
                    else
                        term_tmp *= get_fhat(fhat, j - l, mean)
                    end
                    term += term_tmp
                end
            end
        end

        # special case: if k = 3n, (2n, n) combo is instead a 3-combo and must
        # be counted accordingly. If not, count it simply as another 2-combo
        if k == 3n
            term += 3 * beta(2n, n, lam) * fhat[n]^2
        elseif j_bounds[1] <= 2n <= j_bounds[2] &&
               max(-N, -N + 2n) <= n <= min(N, N + 2n)
            term += 2 * beta(2n, n, lam) * fhat[n] * get_fhat(fhat, k - 2n, mean)
        end
    end

    # single lines
    for j = (-N+k):(N+k)
        l_bounds = (max(-N, -N + j), min(N, N + j))
        # d1, d2 (h, v are held constant)
        # no bounds checking on l since diagonals avoid off-limits areas entirely

        # d1: l = j - n; d2: l = j + n
        if l_bounds[1] <= j - n <= l_bounds[2] &&
           !check_combos((j, j - n), combo_indices)
            term +=
                beta(j, j - n, lam) *
                get_fhat(fhat, k - j, mean) *
                get_fhat(fhat, j - n, mean)
        end
        if l_bounds[1] <= j + n <= l_bounds[2] &&
           !check_combos((j, j + n), combo_indices)
            term +=
                beta(j, j + n, lam) *
                get_fhat(fhat, k - j, mean) *
                get_fhat(fhat, j + n, mean)
        end

        # v1, v2; l = +- n
        if l_bounds[1] <= n <= l_bounds[2] && !check_combos((j, n), combo_indices)
            term +=
                beta(j, n, lam) *
                get_fhat(fhat, k - j, mean) *
                get_fhat(fhat, j - n, mean)
        end
        if l_bounds[1] <= -n <= l_bounds[2] &&
           !check_combos((j, -n), combo_indices)
            term +=
                beta(j, -n, lam) *
                get_fhat(fhat, k - j, mean) *
                get_fhat(fhat, j + n, mean)
        end

        # h1, h2; k - j = n
        if j == k - n || j == k + n
            for l = l_bounds[1]:l_bounds[2]
                if !check_combos((j, l), combo_indices)
                    term +=
                        beta(j, l, lam) *
                        get_fhat(fhat, j - l, mean) *
                        get_fhat(fhat, l, mean)
                end
            end
        end
    end
    return term
end

"""
    construct_jacobian!(jac, fhat, c, N, mean, lam)

Fill the `N × N` matrix `jac` with the Jacobian of the traveling-wave equation
at `fhat`, multithreaded when more than one thread is available.
"""
function construct_jacobian!(
    jac,
    fhat::Vector{ComplexF64},
    c,
    N::Int64,
    mean,
    lam,
)
    if Threads.nthreads() != 1
        @threads :dynamic for n ∈ 1:N
            for k ∈ 1:N
                jac[k, n] = compute_elem(k, n, fhat, c, N, mean, lam)
            end
        end
    else
        for n ∈ 1:N
            for k ∈ 1:N
                jac[k, n] = compute_elem(k, n, fhat, c, N, mean, lam)
            end
        end
    end
end

"""
    F!(output, fhat, c, N, mean, lam)

Evaluate the residual of the traveling-wave equation with speed `c` and base
wavenumber `lam` for Fourier modes `1:N` of `fhat` and store it in `output`.
"""
function F!(output, fhat::Vector{ComplexF64}, c, N::Int64, mean, lam)
    @threads :dynamic for k = 1:N
        sum = zero(ComplexF64)
        for j = (-N+k):(N+k)
            for l = max(-N, -N + j):min(N, N + j)
                sum +=
                    beta(j, l, lam) *
                    get_fhat(fhat, k - j, mean) *
                    get_fhat(fhat, j - l, mean) *
                    get_fhat(fhat, l, mean)
            end
        end
        output[k] = alpha(c, k, lam) * fhat[k] + sum
    end
    return nothing
end

"""
    newton!(sol, guess, c, max_q, mean, lam; tol=1e-11)

Solve the traveling-wave equation for the Fourier coefficients `sol` with
Newton's method, starting from `guess`, for at most `max_q` iterations.

Throw a `ConvergenceError` if the iteration converges to the trivial solution,
blows up, or fails to converge to within `tol`.
"""
function newton!(sol, guess, c, max_q, mean, lam; tol=1e-11)
    N = size(guess, 1)
    tmp = copy(guess)
    jac = zeros(ComplexF64, N, N)
    diff = 0.0
    for _ = 1:max_q
        F!(sol, tmp, c, N, mean, lam)
        construct_jacobian!(jac, tmp, c, N, mean, lam)
        sol .= tmp .- lu!(jac) \ sol
        diff = norm(abs.(sol .- tmp))
        if diff < tol
            if isapprox(sol[1], 0, atol=1e-10)
                throw(ConvergenceError("Trivial solution."))
            else
                return nothing
            end
        elseif diff > 10
            throw(
                ConvergenceError("Blowup; difference between iterations: $diff"),
            )
        end
        tmp .= sol
    end
    throw(
        ConvergenceError(
            "Failed to converge; difference between iterations: $diff",
        ),
    )
end

"""
    print_jac(jac, N, re)

Print the real (`re == true`) or imaginary part of the `N × N` Jacobian `jac`,
with nonzero entries highlighted in green.
"""
function print_jac(jac, N, re::Bool)
    for k = 1:N
        for n = 1:N
            val = re ? real(jac[k, n]) : imag(jac[k, n])
            elem = @sprintf("%+08.3f", val)
            if val != 0
                elem_styled = AnnotatedString(elem, [(1:9, :face, :green)])
                print(elem_styled, "  ")
            else
                print(elem, "  ")
            end
        end
        println()
    end
    println("-----------------------------------------------------------------")
end

struct NovikovProblem
    N_modes::Int64
    c::Float64
    L::Float64
    H::Float64
    xvec::Vector{Float64}
    guess::Vector{Float64}
    guess_hat::Vector{ComplexF64}
    sol::Vector{Float64}
    sol_hat::Vector{ComplexF64}
end

# filesystem struct for reading and writing solutions to files, without
# unnecessary info
struct NovikovSolution
    id::UUID
    c::Float64
    L::Float64
    H::Float64
    sol::Vector{Float64}
    N::Int64
end

"""
    NovikovSolution(p::NovikovProblem)

Convert a `NovikovProblem` into a `NovikovSolution` with a new UUID.
"""
function NovikovSolution(p::NovikovProblem)
    return NovikovSolution(uuid4(), p.c, p.L, p.H, p.sol, length(p.sol))
end

"""
    resize(s::NovikovSolution, res)

Downsample `s` to a fraction `res` of its grid points by keeping every
`1/res`th point. `1/res` must be an even integer, e.g. `res = 1/2`.
"""
function resize(s::NovikovSolution, res)
    try
        res = Int(inv(res))
    catch e
        if e isa InexactError
            throw(
                ArgumentError(
                    "resolution must be a fraction with an even denominator.",
                ),
            )
        else
            error(e)
        end
    end
    res % 2 != 0 && throw(ArgumentError("resolution must have an even inverse."))
    return NovikovSolution(
        s.id,
        s.c,
        s.L,
        s.H,
        s.sol[1:res:end],
        length(1:res:lastindex(s.sol)),
    )
end

"""
    isless(a::NovikovSolution, b::NovikovSolution)

Order solutions by wave speed `c`. Subject to change.
"""
isless(a::NovikovSolution, b::NovikovSolution) = isless(a.c, b.c)

"""
    isequal(a::NovikovSolution, b::NovikovSolution)

Return whether `a` and `b` have the same UUID.
"""
isequal(a::NovikovSolution, b::NovikovSolution) = isequal(a.id, b.id)

struct ConvergenceError <: Exception
    msg::String
end

struct InsufficientModeError <: Exception
    msg::String
    hfm_sum::Float64
end

"""
    showerror(io, err::InsufficientModeError)

Print the error message and the sum of the highest modes.
"""
function Base.showerror(io::IO, err::InsufficientModeError)
    print(io, "InsufficientModeError: ")
    print(io, err.msg * "\nsum of last 15 modes: " * string(err.hfm_sum))
end

"""
    showerror(io, err::ConvergenceError)

Print the error message.
"""
function Base.showerror(io::IO, err::ConvergenceError)
    print(io, "ConvergenceError: ")
    print(io, err.msg)
end


"""
    construct_twsol(guess, c, L; N_gp=1024, N_fs=64, q=20, ctol=1e-12,
                    mtol=1e-12, maxmodes=div(N_gp, 2) + 1)

Compute a traveling-wave solution with speed `c` and period `L` with Newton's
method, starting from the Fourier coefficients `guess` (normalized `rfft`
output; `guess[1]` is the mean).

Newton's method starts with `N_fs` modes; the number of modes is doubled while
the highest modes of the solution sum to more than `mtol`.

# Keyword arguments
- `N_gp`: number of grid points in physical space
- `N_fs`: initial number of Fourier modes used in Newton's method
- `q`: maximum number of Newton iterations
- `ctol`: convergence tolerance of Newton's method
- `mtol`: tolerance on the sum of the highest modes
- `maxmodes`: maximum number of modes; an `InsufficientModeError` is thrown if
  more are needed

Return a `NovikovProblem`. Throw a `ConvergenceError` if Newton's method fails.
"""
function construct_twsol(
    guess::Vector{ComplexF64},
    c::Float64,
    L;
    N_gp=1024,
    N_fs=64,
    q=20,
    ctol=1e-12,
    mtol=1e-12,
    maxmodes=div(N_gp, 2) + 1,
)
    maxmodes < N_fs && throw(
        ArgumentError(
            "Specified maximum number of modes smaller than initial number of modes.",
        ),
    )

    N_fd = div(N_gp, 2) + 1
    lam = 2pi / L
    mean = real(guess[1])

    xvec = collect(0:(N_gp-1)) * L / N_gp
    sol_hat = zeros(ComplexF64, N_fd)
    sol_hat[1] = mean
    sol = Vector{Float64}(undef, N_gp)
    guess_phys = similar(sol)

    iplan = plan_irfft(sol_hat, length(sol))

    # for efficiency, newton's method is only run over the minimum amount of
    # modes to resolve the solution accurately
    newton!(view(sol_hat, 2:N_fs), guess[2:N_fs], c, q, mean, lam, tol=ctol)

    N_fs_new = N_fs

    # increment amount of modes used in NM if highest-frequency modes are higher-
    # valued than desired, as set by mtol
    while real(sum(sol_hat[(N_fs_new-15):N_fs_new])) > mtol
        if N_fs_new * 2 > maxmodes
            throw(
                InsufficientModeError(
                    "Not enough modes to resolve function.",
                    real(sum(sol_hat[(N_fs_new-15):N_fs_new])),
                ),
            )
        else
            N_fs_new *= 2
        end

        log("resized to N=$N_fs_new")
        newton!(
            view(sol_hat, 2:N_fs_new),
            guess[2:N_fs_new],
            c,
            q,
            mean,
            lam,
            tol=ctol,
        )
    end

    if length(guess) == div(length(guess_phys), 2) + 1
        mul!(guess_phys, iplan, guess * N_gp)
    else
        mul!(guess_phys, iplan, vcat(guess, zeros(N_fd - N_fs_new)))
    end

    mul!(sol, iplan, sol_hat * N_gp)
    H = maximum(sol) - minimum(sol)

    return NovikovProblem(
        N_fs_new,
        c,
        L,
        H,
        xvec,
        guess_phys,
        guess,
        sol,
        sol_hat,
    )
end
