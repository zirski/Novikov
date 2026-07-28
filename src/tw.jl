using LinearAlgebra, Printf, StyledStrings, NonlinearSolve, Base.Threads

function alpha(c, n, lam)
    return -c * im * lam * n * (1 + (lam * n)^2)
end

function beta(j, l, lam)
    return im * lam * l * (4 + (lam * l)^2 + 3 * lam^2 * l * (j - l))
end

# Generates combinations of 2 fhat terms; combinations of 3 are used in the diagonal case
function gen_combo_indices(k, n)
    return (
        # flag in last element corresponds to which fhat term is left out (and included in the derivative term):
        # 1: vertical
        # 2: horizontal
        # 3: diagonal
        (k - n, n, 3),       # v1, h1
        (k + n, n, 3),       # v1, h2
        (2n, n, 2),          # v1, d1 -- shares j, l index with 3-combination a1 (k = 3n, j = 2n, l = n)
        (0, n, 2),           # v1, d2
        (k - n, k - 2n, 1),  # h1, d1 
        (k - n, k, 1),       # h1, d2
        (k + n, -n, 3),      # v2, h2
        (k - n, -n, 3),      # v2, h1
        (-2n, -n, 2),        # v2, d2
        (0, -n, 2),          # v2, d1
        (k + n, k + 2n, 1),  # h2, d2
        (k + n, k, 1)        # h2, d1
    )
end

function get_fhat(arr::Vector, i::Int64, mean)
    if i == 0
        return mean
    elseif i < 0
        return arr[-i]
    else
        return arr[i]
    end
end

function compute_elem(k, n, fhat, c, N, mean, lam)
    term = zero(ComplexF64)
    combo_indices = gen_combo_indices(k, n)
    j_bounds = (-N + k, N + k)

    if k == n
        term += alpha(c, n, lam) + 3 * (beta(0, n, lam) + beta(2n, n, lam) + beta(0, -n, lam)) * fhat[n]^2
    else
        # 2 - combinations
        for combo ∈ combo_indices
            if combo != (2n, n, 1) && combo != (2n, n, 2) && combo != (2n, n, 3) # we're not counting the (2n, n) combo since it shares indices with 3-combo and requires special logic
                j = combo[1]
                l = combo[2]
                f = combo[3]
                if j_bounds[1] <= j <= j_bounds[2] && max(-N, -N + j) <= l <= min(N, N + j)
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

        # special case: if k = 3n, (2n, n) combo is instead a 3-combo and must be counted accordingly.
        # if not, count it simply as another 2-combo
        if k == 3n
            term += 3 * beta(2n, n, lam) * fhat[n]^2
        elseif j_bounds[1] <= 2n <= j_bounds[2] && max(-N, -N + 2n) <= n <= min(N, N + 2n)
            term += 2 * beta(2n, n, lam) * fhat[n] * get_fhat(fhat, k - 2n, mean)
        end
    end

    # single lines
    for j = (-N + k):(N + k)
        l_bounds = (max(-N, -N + j), min(N, N + j))
        # d1, d2 (h, v are held constant)
        # IMPORTANT: no bounds checking on l since diagonals avoid off-limits areas entirely

        # d1: l = j - n; d2: l = j + n
        if l_bounds[1] <= j - n <= l_bounds[2] && !((j, j - n, 1) in combo_indices || (j, j - n, 2) in combo_indices || (j, j - n, 3) in combo_indices)
            term += beta(j, j - n, lam) * get_fhat(fhat, k - j, mean) * get_fhat(fhat, j - n, mean)
        end
        if l_bounds[1] <= j + n <= l_bounds[2] && !((j, j + n, 1) in combo_indices || (j, j + n, 2) in combo_indices || (j, j + n, 3) in combo_indices)
            term += beta(j, j + n, lam) * get_fhat(fhat, k - j, mean) * get_fhat(fhat, j + n, mean)
        end

        # v1, v2; l = +- n
        if l_bounds[1] <= n <= l_bounds[2] && !((j, n, 1) in combo_indices || (j, n, 2) in combo_indices || (j, n, 3) in combo_indices)
            term += beta(j, n, lam) * get_fhat(fhat, k - j, mean) * get_fhat(fhat, j - n, mean)
        end
        if l_bounds[1] <= -n <= l_bounds[2] && !((j, -n, 1) in combo_indices || (j, -n, 2) in combo_indices || (j, -n, 3) in combo_indices)
            term += beta(j, -n, lam) * get_fhat(fhat, k - j, mean) * get_fhat(fhat, j + n, mean)
        end

        # h1, h2; k - j = n
        if j == k - n
            for l in l_bounds[1]:l_bounds[2]
                if !((j, l, 1) in combo_indices || (j, l, 2) in combo_indices || (j, l, 3) in combo_indices)
                    term += beta(j, l, lam) * get_fhat(fhat, j - l, mean) * get_fhat(fhat, l, mean)
                end
            end
        end
        if j == k + n
            for l in l_bounds[1]:l_bounds[2]
                if !((j, l, 1) in combo_indices || (j, l, 2) in combo_indices || (j, l, 3) in combo_indices)
                    term += beta(j, l, lam) * get_fhat(fhat, j - l, mean) * get_fhat(fhat, l, mean)
                end
            end
        end
    end
    return term
end

function gen_jacobian_single!(jac, fhat::Vector{ComplexF64}, c, N::Int64, mean, lam)
    for n ∈ 1:N
        for k ∈ 1:N
            jac[k, n] = compute_elem(k, n, fhat, c, N, mean, lam)
        end
    end
end

function gen_jacobian_multi!(jac, fhat::Vector{ComplexF64}, c, N::Int64, mean, lam)
    @threads :dynamic for n ∈ 1:N
        for k ∈ 1:N
            jac[k, n] = compute_elem(k, n, fhat, c, N, mean, lam)
        end
    end
end

function F!(output, fhat::Vector{ComplexF64}, c, N::Int64, mean, lam)
    for k = 1:N
        sum = zero(ComplexF64)
        for j = (-N + k):(N + k)
            for l = max(-N, -N + j):min(N, N + j)
                sum += beta(j, l, lam) * get_fhat(fhat, k - j, mean) * get_fhat(fhat, j - l, mean) * get_fhat(fhat, l, mean)
            end
        end
        output[k] = alpha(c, k, lam) * fhat[k] + sum
    end
    return nothing
end

function newton!(sol, guess, c, L, max_q, mean, lam)
    tol = 1e-11
    N = size(guess, 1)
    tmp = copy(guess)
    jac = zeros(ComplexF64, N, N)
    for _ in 1:max_q
        F!(sol, tmp, c, N, mean, lam)
        gen_jacobian_multi!(jac, tmp, c, N, mean, lam)
        sol .= tmp .- jac \ sol
        diff = norm(abs.(sol .- tmp))
        if diff < tol
            return nothing
        elseif diff > 10
            error("Failed to converge; diff=" * diff)
        end
        tmp .= sol
    end
    errm = string("Failed to converge; diff=", tol)
    error(errm)
end

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

function gen_tw_sol_single(guess::Vector{ComplexF64}, c::Float64, L, N, q, mean=0)
    jac = zeros(ComplexF64, N, N)
    lam = 2pi / L
    fhat = copy(guess)
    F_output = similar(guess)
    fhat_next = zeros(ComplexF64, N)
    norm_tol = 1e-11
    for _ in 1:q
        F!(F_output, fhat, c, N, mean, lam)
        gen_jacobian_single!(jac, fhat, c, N, mean, lam)
        fhat_next .= fhat .- jac \ F_output
        if norm(abs.(fhat .- fhat_next)) < norm_tol
            return fhat_next
        elseif norm(abs.(fhat .- fhat_next)) > 10
            error("Failed to converge")
        end
        fhat .= fhat_next
    end
    return fhat
end

struct NovikovProblem
    N
    xvec
    guess
    guess_hat
    sol
    sol_hat
end

function gen_tw_sol_multi(guess::Vector{ComplexF64}, c::Float64, L; N_gp=1024, q=10)
    N_fd = div(N_gp, 2) + 1
    xvec = collect(1:(N_gp)) * L / (N_gp + 1)
    lam = 2pi / L
    N_rf = size(guess, 1)
    mean = real(guess[1])
    sol_hat = zeros(ComplexF64, N_fd)
    sol_hat[1] = mean
    sol_phys = zeros(N_gp)
    guess_phys = similar(sol_phys)
    iplan = plan_irfft(sol_hat, size(sol_phys, 1))

    newton!(view(sol_hat, 2:N_rf), guess[2:end], c, L, q, mean, lam)    

    # we only need to consider the real component because imaginary part of fourier coefficients
    #  will be zero for all even solutions
    N_rf_new = N_rf
    while real(sol_hat[N_rf_new]) > 1e-12
        N_rf_new *= 2
        println("resized to N=$N_rf_new")
        guess_new = vcat(guess, zeros(ComplexF64, N_rf_new - N_rf))
        # println(size(guess_new))
        newton!(view(sol_hat, 2:N_rf_new), guess_new[2:end], c, L, q, mean, lam)    
    end

    mul!(sol_phys, iplan, sol_hat * N_gp)
    mul!(guess_phys, iplan, vcat(guess, zeros(ComplexF64, N_fd - N_rf)) * N_gp)
    circshift!(sol_phys, div(N_gp, 2))
    circshift!(guess_phys, div(N_gp, 2))
    return NovikovProblem(N_rf_new, xvec, guess_phys, guess, sol_phys, sol_hat)
end
