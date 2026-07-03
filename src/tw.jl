using LinearAlgebra, Printf, NLsolve, StyledStrings

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

function get_fhat(fhat::Vector, i::Int64)
    if i == 0
        return 0
    elseif i < 0
        return fhat[-i]
    else
        return fhat[i]
    end
end

function flip_bit!(coll, idx, n::Int)
    coll[idx] ⊻= (UInt8(1) << n)
end

function gen_jacobian_2(fhat, c, L, N)
    jac = zeros(ComplexF64, N, N)
    elem_terms = zeros(UInt8, 2N + 1)
    lidxs = zeros(Int64, 4)
    bit_indices = [4, 4, 3, 3]

    for d ∈ 1:N
        jac[d, d] = alpha(c, d, L)
    end

    for n ∈ 1:N
        for k ∈ 1:N
            elem = zero(Complex)

            # For each row of double sum, store the l-indices for all terms where k - j, j - l, or l equal n.
            # each term is stored at it's corresponding l-index in elem_terms, where the value is a UInt8 binary sequence.
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
                fill!(elem_terms, 0)
                llb = max(-N, -N + j)
                lrb = min(N, N + j)

                # [v1, v2, d1, d2]
                lidxs[1] = n
                lidxs[2] = -n
                lidxs[3] = j - n
                lidxs[4] = j + n

                for (i, l) in enumerate(lidxs)
                    if llb <= l <= lrb
                        elem_terms[l+N+1] += 1
                        flip_bit!(elem_terms, l + N + 1, bit_indices[i])
                    end
                end

                # h
                if abs(k - j) == n
                    for l ∈ llb:lrb
                        elem_terms[l+N+1] += 1
                        flip_bit!(elem_terms, l + N + 1, 2)
                    end
                end
                # println("Element $k, $n with j = $j")
                # println("-------------matches--------------")
                # for (idx, data) ∈ idx_matches
                #     println("l-index: $idx -->\t", string(data, base=2, pad=8))
                # end
                # println()

                for (aidx, data) ∈ enumerate(elem_terms)
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
    luts = [zeros(Int64, 2N+1, 2N+1) for _ ∈ 1:N, _ ∈ 1:N]
    jac = zeros(ComplexF64, N, N)
    term::ComplexF64 = zero(ComplexF64)
    # main diagonal terms, where k = n = d
    for d in 1:N
        lut = zeros(Int64, 2N + 1, 2N + 1)
        # 3-combinations and alpha term
        term += alpha(c, d, L) + 3 * (beta(0, d, L) + beta(2d, d, L) + beta(0, -d, L)) * fhat[d] ^ 2

        lut[0+N+1-d, d+N+1] += 3
        lut[2d+N+1-d, d+N+1] += 3
        lut[0+N+1-d, -d+N+1] += 3

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
                lut[j+N+1-d, j-d+N+1] += 1
            end
            if l_bounds[1] <= j + d <= l_bounds[2] && !((j, j + d) in combo_indices)
                term += beta(j, j + d, L) * get_fhat(fhat, d-j) * get_fhat(fhat, j+d)
                lut[j+N+1-d, j+d+N+1] += 1
            end

            # v1, v2: l = n, so positions are invariant of j
            # beta * fhat(k - j) * fhat(j - l)
            if l_bounds[1] <= d <= l_bounds[2] && !((j, d) in combo_indices)
                term += beta(j, d, L) * get_fhat(fhat, d-j) * get_fhat(fhat, j-d)
                lut[j+N+1-d, d+N+1] += 1
            end
            if l_bounds[1] <= -d <= l_bounds[2] && !((j, -d) in combo_indices)
                term += beta(j, -d, L) * get_fhat(fhat, d-j) * get_fhat(fhat, j+d)
                lut[j+N+1-d, -d+N+1] += 1
            end

            # h1, h2: we can simplify the condition j = k - n, j = k + n since k = n
            if j == 0
                for l ∈ l_bounds[1]:l_bounds[2]
                    if !((j, l) in combo_indices)
                        term += beta(j, l, L) * get_fhat(fhat, l) * get_fhat(fhat, l)
                        lut[j+N+1-d, l+N+1] += 1
                    end
                end
            end
            if j == 2d
                for l ∈ l_bounds[1]:l_bounds[2]
                    if !((j, l) in combo_indices)
                        term += beta(j, l, L) * get_fhat(fhat, j-l) * get_fhat(fhat, l)
                        lut[j+N+1-d, l+N+1] += 1
                    end
                end
            end
        end
        jac[d, d] = term
        # println("Equation $d for unknown $d:")
        # for i ∈ 1:(2N+1)
        #     for j ∈ 1:(2N+1)
        #         val = lut[i, j]
        #         if val != 0
        #             val_str = AnnotatedString(string(val), [(1:9, :face, :green)])
        #             print(val_str, "  ")
        #         elseif j - N - 1 < -N + i - N - 1 + d || j - N - 1 > N + i - N - 1 + d
        #             val_str = AnnotatedString(string(val), [(1:9, :face, :red)])
        #             print(val_str, "  ")
        #         else
        #             print(lut[i, j], "  ")
        #         end
        #     end
        #     println()
        #     println()
        # end
        # println()
        luts[d, d] .= lut
        term = zero(Complex)
    end
    term = zero(Complex)

    # all other terms
    for k = 1:N
        for n = 1:N
            if k != n
                lut = zeros(Int64, 2N + 1, 2N + 1)

                combo_indices = gen_combo_indices(k, n)
                j_bounds = (-N + k, N + k)

                # 2 - combinations
                for (i, combo) ∈ enumerate(combo_indices)
                    if combo != (2n, n, 1) && combo != (2n, n, 2) && combo != (2n, n, 3) # we're not counting the (2n, n) combo since it shares indices with 3-combo and requires special logic
                        j = combo[1]
                        l = combo[2]
                        f = combo[3]
                        term_tmp = 2 * beta(j, l, L) * fhat[n]
                        if j_bounds[1] <= j <= j_bounds[2]
                            if max(-N, -N + j) <= l <= min(N, N + j)
                                lut[j+N+1-k, l+N+1] += 2
                                if f == 1
                                    term_tmp *= get_fhat(fhat, l)
                                elseif f == 2
                                    term_tmp *= get_fhat(fhat, k-j)
                                else
                                    term_tmp *= get_fhat(fhat, j-l)
                                end
                            end
                        end
                        term += term_tmp
                    end
                end

                # special case: if k = 3n, (2n, n) combo is instead a 3-combo and must be counted accordingly.
                # if not, count it simply as another 2-combo
                if k == 3n
                    term += 3 * beta(2n, n, L) * fhat[n] ^ 2
                    lut[2n+N+1-k, n+N+1] += 3
                elseif j_bounds[1] <= 2n <= j_bounds[2] && max(-N, -N + 2n) <= n <= min(N, N + 2n)
                    term += 2 * beta(2n, n, L) * fhat[n] * get_fhat(fhat, k - n)
                    lut[2n+N+1-k, n+N+1] += 2
                end

                # single lines
                for j = (-N+k):(N+k)
                    l_bounds = (max(-N, -N+j), min(N, N+j))
                    # d1, d2 (h, v are held constant)
                    # IMPORTANT: no bounds checking on l since diagonals avoid off-limits areas entirely

                    # d1: l = j - n; d2: l = j + n
                    if l_bounds[1] <= j - n <= l_bounds[2] && !((j, j - n, 1) in combo_indices || (j, j - n, 2) in combo_indices || (j, j - n, 3) in combo_indices)
                        lut[j+N+1-k, j-n+N+1] += 1
                        term += beta(j, j - n, L) * get_fhat(fhat, k-j) * get_fhat(fhat, j-n)
                    end
                    if l_bounds[1] <= j + n <= l_bounds[2] && !((j, j + n, 1) in combo_indices || (j, j + n, 2) in combo_indices || (j, j + n, 3) in combo_indices)
                        lut[j+N+1-k, j+n+N+1] += 1
                        term += beta(j, j + n, L) * get_fhat(fhat, k-j) * get_fhat(fhat, j+n)
                    end

                    # v1, v2; l = +- n
                    if l_bounds[1] <= n <= l_bounds[2] && !((j, n, 1) in combo_indices || (j, n, 2) in combo_indices || (j, n, 3) in combo_indices)
                        lut[j+N+1-k, n+N+1] += 1
                        term += beta(j, n, L) * get_fhat(fhat, k-j) * get_fhat(fhat, j-n)
                    end
                    if l_bounds[1] <= -n <= l_bounds[2] && !((j, -n, 1) in combo_indices || (j, -n, 2) in combo_indices || (j, -n, 3) in combo_indices)
                        lut[j+N+1-k, -n+N+1] += 1
                        term += beta(j, -n, L) * get_fhat(fhat, k-j) * get_fhat(fhat, j+n)
                    end

                    # h1, h2; k - j = n
                    if j == k - n
                        for l in l_bounds[1]:l_bounds[2]
                            if !((j, l, 1) in combo_indices || (j, l, 2) in combo_indices || (j, l, 3) in combo_indices)
                                lut[j+N+1-k, l+N+1] += 1
                                term += beta(j, l, L) * get_fhat(fhat, j-l) * get_fhat(fhat, l)
                            end
                        end
                    end
                    if j == k + n
                        for l in l_bounds[1]:l_bounds[2]
                            if !((j, l, 1) in combo_indices || (j, l, 2) in combo_indices || (j, l, 3) in combo_indices)
                                lut[j+N+1-k, l+N+1] += 1
                                term += beta(j, l, L) * get_fhat(fhat, j-l) * get_fhat(fhat, l)
                            end
                        end
                    end
                end
                jac[k, n] = term
                # println("Equation $k for unknown $n:")
                # for i ∈ 1:(2N+1)
                #     for j ∈ 1:(2N+1)
                #         val = lut[i, j]
                #         if val != 0
                #             val_str = AnnotatedString(string(val), [(1:9, :face, :green)])
                #             print(val_str, "  ")
                #         elseif j - N - 1 < -N + i - N - 1 + k || j - N - 1 > N + i - N - 1 + k
                #             val_str = AnnotatedString(string(val), [(1:9, :face, :red)])
                #             print(val_str, "  ")
                #         else
                #             print(lut[i, j], "  ")
                #         end
                #     end
                #     println()
                #     println()
                # end
                # println()
                luts[k, n] .= lut
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
            for l = max(-N, -N+j):min(N, N+j)
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
    norm_tol = 1e-11
    norm(fhat .- fhat_next)
    for _ in 1:q
        f = F(fhat, c, L, N)
        jac = gen_jacobian_2(fhat, c, L, N)
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
    norm_tol = 1e-11
    norm(fhat .- fhat_next)
    for _ in 1:q
        f = F(fhat, c, L, N)
        jac = gen_jacobian_1(fhat, c, L, N)
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