using Revise

function gen_eq(k, N)
    index_matrix = Matrix{Tuple{Int,Int,Int}}(undef, 2N + 1, 2N + 1)
    for i = 1:2N+1
        for j = 1:2N+1
            index_matrix[i, j] = (0, 0, 0)
        end
    end

    for j = max(-2N, -N + k):min(2N, N + k)
        for l = max(-N, -N + j):min(N, N + j)
            r = j + N + 1
            c = l + N + 1
            index_matrix[r, c] = (k - j, j - l, l)
        end
    end
    return index_matrix
end

function print_eq(m, k, N)
    m_size = 2N + 1
    print("     ")
    for i = 1:m_size
        print(i, "  ")
    end
    println()
    print("    ")
    for i = 1:m_size
        print(i - N - 1, " ")
        if i - N >= 0
            print(" ")
        end
    end
    println()
    println("    -------------l------------")
    for i = 1:m_size
        print(i)
        if i - N - 1 >= 0
            print(" ")
        end
        print(i - N - 1)
        print("|", " ")
        for j = 1:m_size
            matches = 0

            if m[i, j][1] == m[i, j][2] && m[i, j][1] == k
                matches += 1
            end
            if m[i, j][1] == m[i, j][3] && m[i, j][1] == k
                matches += 1
            end
            if m[i, j][2] == m[i, j][3] && m[i, j][2] == k
                matches += 1
            end
            print(matches, "  ")
        end
        println()
        println("   |")
    end
end

k = 1
N = 4
m = gen_eq(k, N)
print_eq(m, k, N)

