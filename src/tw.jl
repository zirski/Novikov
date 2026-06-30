using LinearAlgebra, Plots, Printf, NLsolve

function alpha(c, n, L)
    lam = 2 * pi / L
    return -c * im * lam * n * (1 + (lam * n)^2)
end

function beta(j, l, L)
    lam = 2 * pi / L
    return im * lam * l * (4 + (lam * l)^2 + 3 * lam ^ 2 * l * (j - l))
end

function gen_combo_indices(k, n)
    return (
        # flag in last element corresponds to which fhat term is left out (and included in the derivative term):
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
        (k + n, k, 1)        # h2, d1
    )
end

function get_fhat(fhat::Vector, i::Int64)
    if i == 0
        return 0
    elseif i < 0
        return fhat[-i]
    else
        return fhat[i]
    end
end

function flip_bit!(dict, key, n::Int)
    dict[key] ⊻= (UInt8(1) << n)
end

function gen_jacobian_2(fhat, c, L, N)
    jac = zeros(ComplexF64, N, N)
    idx_matches = Dict{Int64,UInt8}()
    paths = zeros(Int64, 4)
    bit_indices = [4, 4, 3, 3]

    for d ∈ 1:N
        jac[d, d] = alpha(c, d, L)
    end

    for n ∈ 1:N
        for k ∈ 1:N
            elem = zero(Complex)
            for j ∈ (-N+k):(N+k)
                # flush dictionary
                for k in keys(idx_matches)
                    idx_matches[k] = 0
                end
                # println(j)
                llb = max(-N, -N + j)
                lrb = min(N, N + j)

                # adds all l-indices for a specific j-row to a dictionary, where 
                # keys correspond to indices and values represent how many lines 
                # "hit" the index

                # [v1, v2, d1, d2]
                paths .= [n, -n, j - n, j + n]

                for (i, path) in enumerate(paths)
                    if llb <= path <= lrb
                        if haskey(idx_matches, path)
                            idx_matches[path] += 1
                        else
                            idx_matches[path] = 1
                        end
                        flip_bit!(idx_matches, path, bit_indices[i])
                    end
                end


                # # v1
                # if llb <= n <= lrb
                #     idx_matches[n] = get(idx_matches, n, 0) + 1
                #     idx_matches[n] = flip_bit(idx_matches[n], 4)
                #     # println("added v1: l = $n")
                # end
                # # v2
                # if llb <= -n <= lrb
                #     idx_matches[-n] = get(idx_matches, -n, 0) + 1
                #     idx_matches[-n] = flip_bit(idx_matches[-n], 4)
                #     # println("added v2: l = $(-n)")
                # end
                # # d1
                # if llb <= j - n <= lrb
                #     idx_matches[j-n] = get(idx_matches, j - n, 0) + 1
                #     idx_matches[j-n] = flip_bit(idx_matches[j-n], 3)
                #     # println("added d1: l = $(j - n)")
                # end
                # # d2
                # if llb <= j + n <= lrb
                #     idx_matches[j+n] = get(idx_matches, j + n, 0) + 1
                #     idx_matches[j+n] = flip_bit(idx_matches[j+n], 3)
                #     # println("added d2: l = $(j + n)")
                # end
                # # h
                if abs(k - j) == n
                    for idx ∈ llb:lrb
                        if haskey(idx_matches, idx)
                            idx_matches[idx] += 1
                        else
                            idx_matches[idx] = 1
                        end
                        flip_bit!(idx_matches, idx, 2)
                    end
                end
                # println("Element $k, $n with j = $j")
                # println("-------------matches--------------")
                # for (idx, data) ∈ idx_matches
                #     println("l-index: $idx -->\t", string(data, base=2, pad=8))
                # end
                # println()

                for (idx, data) ∈ idx_matches
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
                        # println(order)

                        temp_elem += order * beta(j, idx, L) * fhat[n] ^ (order - 1)
                        if h
                            temp_elem *= get_fhat(fhat, k - j)
                        end
                        if d
                            temp_elem *= get_fhat(fhat, j - idx)
                        end
                        if v
                            temp_elem *= get_fhat(fhat, idx)
                        end
                        elem += temp_elem
                    end
                end
            end
            # println(elem)
            jac[k, n] += elem
        end
    end
    # println(jac)
    return jac
end

function gen_jacobian_1(fhat::Vector{ComplexF64}, c, L::Float64, N::Int64)
    jac = zeros(ComplexF64, N, N)
    term::ComplexF64 = zero(ComplexF64)
    # main diagonal terms, where k = n = d
    for d in 1:N
        # 3-combinations and alpha term
        term += alpha(c, d, L) + 3 * (beta(0, d, L) + beta(2d, d, L) + beta(0, -d, L)) * fhat[d] ^ 2
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
                term += beta(j, j - d, L) * get_fhat(fhat, d-j) * get_fhat(fhat, j-d)
            end
            if l_bounds[1] <= j + d <= l_bounds[2] && !((j, j + d) in combo_indices)
                term += beta(j, j + d, L) * get_fhat(fhat, d-j) * get_fhat(fhat, j+d)
            end

            # v1, v2: l = n, so positions are invariant of j
            # beta * fhat(k - j) * fhat(j - l)
            if l_bounds[1] <= d <= l_bounds[2] && !((j, d) in combo_indices)
                term += beta(j, d, L) * get_fhat(fhat, d-j) * get_fhat(fhat, j-d)
            end
            if l_bounds[1] <= -d <= l_bounds[2] && !((j, -d) in combo_indices)
                term += beta(j, -d, L) * get_fhat(fhat, d-j) * get_fhat(fhat, j+d)
            end

            # h1, h2: we can simplify the condition j = k - n, j = k + n since k = n
            if j == 0
                for l in l_bounds
                    if !((j, l) in combo_indices)
                        term += beta(j, l, L) * get_fhat(fhat, l) * get_fhat(fhat, l)
                    end
                end
            end
            if j == 2d
                for l in l_bounds
                    if !((j, l) in combo_indices)
                        term += beta(j, l, L) * get_fhat(fhat, j-l) * get_fhat(fhat, l)
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

                if k == 3n
                    # a1
                    term += 3 * beta(2n, n, L) * fhat[n] ^ 2
                    # 2 - combinations
                    for combo in combo_indices
                        j = combo[1]
                        l = combo[2]
                        term_tmp = 2 * beta(j, l, L) * fhat[n]
                        if combo != (2n, n) && j_bounds[1] <= j <= j_bounds[2]
                            if max(-N, -N + j) <= l <= min(N, N + j)
                                if combo[3] == 1
                                    term_tmp *= get_fhat(fhat, l)
                                elseif combo[3] == 2
                                    term_tmp *= get_fhat(fhat, k-j)
                                else
                                    term_tmp *= get_fhat(fhat, j-l)
                                end
                            end
                        end
                        term += term_tmp
                    end
                else
                    for combo in combo_indices
                        j = combo[1]
                        l = combo[2]
                        term_tmp = 2 * beta(j, l, L) * fhat[n]
                        if j_bounds[1] <= j <= j_bounds[2]
                            if max(-N, -N + j) <= l <= min(N, N + j)
                                if combo[3] == 1
                                    term_tmp *= get_fhat(fhat, l)
                                elseif combo[3] == 2
                                    term_tmp *= get_fhat(fhat, k-j)
                                else
                                    term_tmp *= get_fhat(fhat, j-l)
                                end
                            end
                        end
                        term += term_tmp
                    end
                end

                # single lines
                for j = (-N+k):(N+k)
                    l_bounds = (max(-N, -N+j), min(N, N+j))
                    # d1, d2 (h, v are held constant)
                    # IMPORTANT: no bounds checking on l since diagonals avoid off-limits areas entirely

                    # d1: l = j - n; d2: l = j + n
                    if l_bounds[1] <= j - n <= l_bounds[2] && !((j, j - n) in combo_indices)
                        term += beta(j, j - n, L) * get_fhat(fhat, k-j) * get_fhat(fhat, j-n)
                    end
                    if l_bounds[1] <= j + n <= l_bounds[2] && !((j, j + n) in combo_indices)
                        term += beta(j, j + n, L) * get_fhat(fhat, k-j) * get_fhat(fhat, j+n)
                    end

                    # v1, v2; l = +- n
                    if l_bounds[1] <= n <= l_bounds[2] && !((j, n) in combo_indices)
                        term += beta(j, n, L) * get_fhat(fhat, k-j) * get_fhat(fhat, j-n)
                    end
                    if l_bounds[1] <= -n <= l_bounds[2] && !((j, -n) in combo_indices)
                        term += beta(j, -n, L) * get_fhat(fhat, k-j) * get_fhat(fhat, j+n)
                    end

                    # h1, h2; k - j = n
                    if j == k - n
                        for l in l_bounds
                            if !((j, l) in combo_indices)
                                term += beta(j, l, L) * get_fhat(fhat, j-l) * get_fhat(fhat, l)
                            end
                        end
                    end
                    if j == k + n
                        for l in l_bounds
                            if !((j, l) in combo_indices)
                                term += beta(j, l, L) * get_fhat(fhat, j-l) * get_fhat(fhat, l)
                            end
                        end
                    end
                end
                jac[k, n] = term
            end
            term = zero(Complex)
        end
    end
    return jac
end

function F(fhat::Vector{ComplexF64}, c, L, N)
    sol = zeros(ComplexF64, N)
    for k = 1:N
        sum = zero(ComplexF64)
        for j = (-N+k):(N+k)
            for l = (max(-N, -N+j), min(N, N+j))
                sum += beta(j, l, L) * get_fhat(fhat, k - j) * get_fhat(fhat, j - l) * get_fhat(fhat, l)
            end
        end
        sol[k] = alpha(c, k, L) * fhat[k] + sum
    end
    return sol
end

function print_jac(jac, N, re::Bool)
    for n = 1:N
        for k = 1:N
            if re
                elem = @sprintf("%+4.5f", real(jac[k, n]))
                print(elem, "\t")
            else
                elem = @sprintf("%+4.5f", imag(jac[k, n]))
                print(elem, "\t")
            end
        end
        println()
    end
    println("-----------------------------------------------------------------")
end

function gen_tw_sol_3(guess::Vector{ComplexF64}, c::Float64, L, N, q)
    fhat = guess
    fhat_next = zeros(ComplexF64, N)
    c_next = c
    norm_tol = 1e-11
    norm(fhat .- fhat_next)
    for _ in 1:q
        f = F(fhat, c_next, L, N)
        jac = gen_jacobian_2(fhat, c_next, L, N)
        # print_jac(jac, N, false)
        fhat_next = fhat .- jac \ f
        if norm(abs.(fhat .- fhat_next)) < norm_tol
            return fhat_next
        end
        println(norm(abs.(fhat .- fhat_next)))
        fhat = fhat_next
    end
    return fhat
end

function gen_tw_sol_2(guess::Vector{ComplexF64}, c::Float64, L, N, q)
    fhat = guess
    fhat_next = zeros(ComplexF64, N)
    c_next = c
    norm_tol = 1e-11
    norm(fhat .- fhat_next)
    for _ in 1:q
        f = F(fhat, c_next, L, N)
        jac = gen_jacobian_1(fhat, c_next, L, N)
        # print_jac(jac, N, false)
        fhat_next = fhat .- jac \ f
        if norm(abs.(fhat .- fhat_next)) < norm_tol
            return fhat_next
        end
        println(norm(abs.(fhat .- fhat_next)))
        fhat = fhat_next
    end
    return fhat
end

function gen_tw_sol_1(guess::Vector{ComplexF64}, c::Float64, L, N, q)
    guess_tmp = copy(guess)
    F_s(x) = F(x, c, L, N - 1)
    sol = nlsolve(F_s, guess_tmp[2:end], iterations=q)
    guess_tmp[2:end] .= sol.zero
    return guess_tmp
end