using LinearAlgebra, Printf, StyledStrings, NonlinearSolve

function alpha(c, n, lam)
    return -c * im * lam * n * (1 + (lam * n)^2)
end

function beta(j, l, lam)
    return im * lam * l * (4 + (lam * l) ^ 2 + 3 * lam ^ 2 * l * (j - l))
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

function flip_bit!(coll, idx, n::Int)
    coll[idx] ⊻= (UInt8(1) << n)
end

function gen_jacobian_2!(jac, fhat, term_vec, c, N, mean, lam)
    lidxs = zeros(Int64, 4)
    bit_indices = [4, 4, 3, 3]

    for d ∈ 1:N
        jac[d, d] = alpha(c, d, lam)
    end

    for n ∈ 1:N
        for k ∈ 1:N
            # println("k = $k, n = $n")
            elem = zero(Complex)

            # For each row of double sum, store the l-indices for all terms where k - j, j - l, and/or l equal n.
            # each term is stored at it's corresponding l-index in term_vec, where the value is a UInt8 binary sequence.
            #
            # value looks like:     0 0 0 1 1 0 1 0
            #                       7 6 5 4 3 2 1 0
            #                      |----- ----| ---|
            #                      |          |    |
            #                  unused bits    |    These bits are used to store the number of fhat arguments which match n
            #                                 |
            #                                 These bits store which fhat args equal n; the 0s indicate which leftover fhats should be included in the final derivative
            # In the above example, this term looks like:   beta(j, l, L) * fhat(n)^2 * fhat(l)
            for j ∈ (-N+k):(N+k)
                fill!(term_vec, 0)
                llb = max(-N, -N + j)
                lrb = min(N, N + j)

                # [v1, v2, d1, d2]
                lidxs[1] = n
                lidxs[2] = -n
                lidxs[3] = j - n
                lidxs[4] = j + n

                for (i, l) in enumerate(lidxs)
                    if llb <= l <= lrb
                        term_vec[l+N+1] += 1
                        flip_bit!(term_vec, l + N + 1, bit_indices[i])
                    end
                end

                # h
                if abs(k - j) == n
                    for l ∈ llb:lrb
                        term_vec[l+N+1] += 1
                        flip_bit!(term_vec, l + N + 1, 2)
                    end
                end
                # println("Element $k, $n with j = $j")
                # println("-------------matches--------------")
                # for (idx, data) ∈ enumerate(term_vec)
                #     println("l-index: $(idx - N - 1) -->\t", string(data, base=2, pad=8))
                # end
                # println()

                for (aidx, data) ∈ enumerate(term_vec)
                    # mapping from julia 1-indexing to -N:N indexing
                    idx = aidx - N - 1
                    if data != 0
                        temp_elem = zero(Complex)
                        # println("l-index: $idx")
                        h = !Bool((data >> 2) & 1)
                        # println(h)
                        d = !Bool((data >> 3) & 1)
                        # println(d)
                        v = !Bool((data >> 4) & 1)
                        # println(v)

                        order = data & 3
                        # print(order, "  ")

                        temp_elem = order * beta(j, idx, lam) * fhat[n] ^ (order - 1)
                        if h
                            temp_elem *= get_fhat(fhat, k - j, mean)
                        end
                        if d
                            temp_elem *= get_fhat(fhat, j - idx, mean)
                        end
                        if v
                            temp_elem *= get_fhat(fhat, idx, mean)
                        end
                        elem += temp_elem
                    else
                        # print(0, "  ")
                    end
                end
                # println()
            end
            # println(elem)
            if k == n
                jac[k, n] += elem
            else
                jac[k, n] = elem
            end
            # println("----------------------------------")
        end
    end
    # println(jac)
end

function gen_jacobian_1(fhat::Vector{ComplexF64}, c, N::Int64, mean, lam)
    jac = zeros(ComplexF64, N, N)
    term::ComplexF64 = zero(ComplexF64)
    # main diagonal terms, where k = n = d
    for d in 1:N
        # 3-combinations and alpha term
        term += alpha(c, d, lam) + 3 * (beta(0, d, lam) + beta(2d, d, lam) + beta(0, -d, lam)) * fhat[d] ^ 2
        combo_indices = ((0, d), (2d, d), (0, -d))
        # line segments:
        # We add terms for each j "row", checking if the l-indices are in-bounds for each line.
        for j = (-N+d):(N+d)
            l_bounds = (max(-N, -N+j), min(N, N+j))

            # d1, d2 (h, v are held constant)
            # we can treat the diagonals as linear functions of j (which we know) to compute the l

            # d1: l = j - n; d2: l = j + n
            # beta * fhat(k - j) * fhat(l)
            if l_bounds[1] <= j - d <= l_bounds[2] && !((j, j - d) in combo_indices)
                term += beta(j, j - d, lam) * get_fhat(fhat, d-j, mean) * get_fhat(fhat, j-d, mean)
            end
            if l_bounds[1] <= j + d <= l_bounds[2] && !((j, j + d) in combo_indices)
                term += beta(j, j + d, lam) * get_fhat(fhat, d-j, mean) * get_fhat(fhat, j+d, mean)
            end

            # v1, v2: l = n, so positions are invariant of j
            # beta * fhat(k - j) * fhat(j - l)
            if l_bounds[1] <= d <= l_bounds[2] && !((j, d) in combo_indices)
                term += beta(j, d, lam) * get_fhat(fhat, d-j, mean) * get_fhat(fhat, j-d, mean)
            end
            if l_bounds[1] <= -d <= l_bounds[2] && !((j, -d) in combo_indices)
                term += beta(j, -d, lam) * get_fhat(fhat, d-j, mean) * get_fhat(fhat, j+d, mean)
            end

            # h1, h2: we can simplify the condition j = k - n, j = k + n since k = n
            if j == 0
                for l ∈ l_bounds[1]:l_bounds[2]
                    if !((j, l) in combo_indices)
                        term += beta(j, l, lam) * get_fhat(fhat, l, mean) * get_fhat(fhat, l, mean)
                    end
                end
            end
            if j == 2d
                for l ∈ l_bounds[1]:l_bounds[2]
                    if !((j, l) in combo_indices)
                        term += beta(j, l, lam) * get_fhat(fhat, j-l, mean) * get_fhat(fhat, l, mean)
                    end
                end
            end
        end
        jac[d, d] = term
        term = zero(Complex)
    end
    term = zero(Complex)

    # all other terms
    for k = 1:N
        for n = 1:N
            if k != n
                combo_indices = gen_combo_indices(k, n)
                j_bounds = (-N + k, N + k)

                # 2 - combinations
                for combo ∈ combo_indices
                    if combo != (2n, n, 1) && combo != (2n, n, 2) && combo != (2n, n, 3) # we're not counting the (2n, n) combo since it shares indices with 3-combo and requires special logic
                        j = combo[1]
                        l = combo[2]
                        f = combo[3]
                        term_tmp = 2 * beta(j, l, lam) * fhat[n]
                        if j_bounds[1] <= j <= j_bounds[2]
                            if max(-N, -N + j) <= l <= min(N, N + j)
                                if f == 1
                                    term_tmp *= get_fhat(fhat, l, mean)
                                elseif f == 2
                                    term_tmp *= get_fhat(fhat, k-j, mean)
                                else
                                    term_tmp *= get_fhat(fhat, j-l, mean)
                                end
                            end
                        end
                        term += term_tmp
                    end
                end

                # special case: if k = 3n, (2n, n) combo is instead a 3-combo and must be counted accordingly.
                # if not, count it simply as another 2-combo
                if k == 3n
                    term += 3 * beta(2n, n, lam) * fhat[n] ^ 2
                elseif j_bounds[1] <= 2n <= j_bounds[2] && max(-N, -N + 2n) <= n <= min(N, N + 2n)
                    term += 2 * beta(2n, n, lam) * fhat[n] * get_fhat(fhat, k - n, mean)
                end

                # single lines
                for j = (-N+k):(N+k)
                    l_bounds = (max(-N, -N+j), min(N, N+j))
                    # d1, d2 (h, v are held constant)
                    # IMPORTANT: no bounds checking on l since diagonals avoid off-limits areas entirely

                    # d1: l = j - n; d2: l = j + n
                    if l_bounds[1] <= j - n <= l_bounds[2] && !((j, j - n, 1) in combo_indices || (j, j - n, 2) in combo_indices || (j, j - n, 3) in combo_indices)
                        term += beta(j, j - n, lam) * get_fhat(fhat, k-j, mean) * get_fhat(fhat, j-n, mean)
                    end
                    if l_bounds[1] <= j + n <= l_bounds[2] && !((j, j + n, 1) in combo_indices || (j, j + n, 2) in combo_indices || (j, j + n, 3) in combo_indices)
                        term += beta(j, j + n, lam) * get_fhat(fhat, k-j, mean) * get_fhat(fhat, j+n, mean)
                    end

                    # v1, v2; l = +- n
                    if l_bounds[1] <= n <= l_bounds[2] && !((j, n, 1) in combo_indices || (j, n, 2) in combo_indices || (j, n, 3) in combo_indices)
                        term += beta(j, n, lam) * get_fhat(fhat, k-j, mean) * get_fhat(fhat, j-n, mean)
                    end
                    if l_bounds[1] <= -n <= l_bounds[2] && !((j, -n, 1) in combo_indices || (j, -n, 2) in combo_indices || (j, -n, 3) in combo_indices)
                        term += beta(j, -n, lam) * get_fhat(fhat, k-j, mean) * get_fhat(fhat, j+n, mean)
                    end

                    # h1, h2; k - j = n
                    if j == k - n
                        for l in l_bounds[1]:l_bounds[2]
                            if !((j, l, 1) in combo_indices || (j, l, 2) in combo_indices || (j, l, 3) in combo_indices)
                                term += beta(j, l, lam) * get_fhat(fhat, j-l, mean) * get_fhat(fhat, l, mean)
                            end
                        end
                    end
                    if j == k + n
                        for l in l_bounds[1]:l_bounds[2]
                            if !((j, l, 1) in combo_indices || (j, l, 2) in combo_indices || (j, l, 3) in combo_indices)
                                term += beta(j, l, lam) * get_fhat(fhat, j-l, mean) * get_fhat(fhat, l, mean)
                            end
                        end
                    end
                end
                jac[k, n] = term
                term = zero(Complex)
            end
        end
    end
    return jac
end

function F(fhat::Vector{ComplexF64}, c, N::Int64, mean, lam)
    sol = zeros(ComplexF64, N)
    for k = 1:N
        sum = zero(ComplexF64)
        for j = (-N+k):(N+k)
            for l = max(-N, -N+j):min(N, N+j)
                sum += beta(j, l, lam) * get_fhat(fhat, k - j, mean) * get_fhat(fhat, j - l, mean) * get_fhat(fhat, l, mean)
            end
        end
        sol[k] = alpha(c, k, lam) * fhat[k] + sum
    end
    return sol
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

function gen_tw_sol_3(guess::Vector{ComplexF64}, c::Float64, L, N, q, mean=0)
    jac = zeros(ComplexF64, N, N)
    term_vec = zeros(UInt8, 2N + 1)
    lam = 2pi / L
    fhat = copy(guess)
    fhat_next = zeros(ComplexF64, N)
    norm_tol = 1e-11
    norm(fhat .- fhat_next)
    for _ in 1:q
        f = F(fhat, c, N, mean, lam)
        gen_jacobian_2!(jac, fhat, term_vec, c, N, mean, lam)
        fhat_next .= fhat .- jac \ f
        if norm(abs.(fhat .- fhat_next)) < norm_tol
            return fhat_next
        end
        fhat .= fhat_next
    end
    return fhat_next
end

function gen_tw_sol_2(guess::Vector{ComplexF64}, c::Float64, L, N, q, mean=0)
    lam = 2pi / L
    fhat = copy(guess)
    fhat_next = zeros(ComplexF64, N)
    norm_tol = 1e-11
    norm(fhat .- fhat_next)
    for _ in 1:q
        f = F(fhat, c, N, mean, lam)
        jac = gen_jacobian_1(fhat, c, N, mean, lam)
        fhat_next .= fhat .- jac \ f
        if norm(abs.(fhat .- fhat_next)) < norm_tol
            return fhat_next
            # elseif norm(abs.(fhat .- fhat_next)) > 100
            #     error("Failed to converge")
        end
        println(norm(abs.(fhat .- fhat_next)))
        fhat .= fhat_next
    end
    return fhat
end

function gen_tw_sol_nonlinear(guess::Vector{ComplexF64}, c::Float64, L, N::Int64, mean=0)
    lam = 2pi / L
    p = (c, N, mean, lam)
    guess_tmp = copy(guess)
    # NonLinearSolve needs the model function to be of the form f(u, p) where p is a collection of
    # parameters; so we need to first package the arguments to the F function defined above in an array
    # and define the function F_s to accept p as its second argument to fit the structure accepted by NLS.
    F_s(x, p) = F(x, p[1], p[2], p[3], p[4])
    prob = NonlinearProblem(F_s, guess_tmp, p)
    sol = solve(prob)
    guess_tmp .= sol.u
    return guess_tmp
end
