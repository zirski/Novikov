using LinearAlgebra

function beta(j, l, L)
    lam = 2 * pi / L
    return im * lam * (4 + (lam * l)^2) - 3 * lam * (j - l) * (lam * l)^3
end

function alpha(c, n, L)
    lam = 2 * pi / L
    return -c * (im * lam * n + (lam * n)^3)
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

function get_fhat(fhat::Vector, i::Integer)
    if i == 0
        return 0
    elseif i < 0
        return fhat[-i]
    else
        return fhat[i]
    end
end

function gen_jacobian(fhat::Vector{ComplexF64}, c, L::Float64, N::Integer)
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
    end
    term = zero(Complex)

    # all other terms
    for n = 1:N
        for k = 1:N
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
        end
    end

    return jac
end

function gen_tw_sol(c::Float64, guess::Vector{ComplexF64})

end