using Revise, StyledStrings

function print_eq(k, N, flag, n)
    # creates matrix, populates with 0s
    m = Matrix{Tuple{Int,Int,Int}}(undef, 2N + 1, 2N + 1)
    for i = 1:(2N+1)
        for j = 1:(2N+1)
            m[i, j] = (0, 0, 0)
        end
    end

    # computes fhat arguments for each term in sum based on equation (6) from 
    # notes document
    # original bounds for j were max(-2N, -N + k):min(2N, N + k) to account for
    # negative k; we simplify for conciseness
    for j = (-N+k):(N+k)
        for l = max(-N, -N+j):min(N, N+j)
            r = j + N + 1 - k
            c = l + N + 1
            m[r, c] = (k - j, j - l, l)
        end
    end

    # graph header and horizontal axis setup
    println("N = ", N)
    println("k = ", k)
    println("n = ", n)
    m_size = 2N + 1
    print("      ")

    # first prints julia matrix indices
    for i = 1:m_size
        print(i, " ")
        if i < 10
            print(" ")
        end
    end
    println()
    print("     ")

    # then prints indices which correspond to the actual j/l index in the sum
    for i = 1:m_size
        print(i - N - 1, " ")
        if i - N >= 0
            print(" ")
        end
    end
    println()

    # main graph printing
    println("    -------------l------------")
    for i = 1:m_size

        # vertical axis setup
        print(i)
        if i - N - 1 + k >= 0
            print(" ")
        end
        if i < 10
            print(" ")
        end
        print(i - N - 1 + k)
        print("|", " ")

        # what the graph outputs: "flag" argument specifies:
        # 0: print if term contains fhat(index), where index is the desired fhat(k)
        # 1 - 3: print the argument for fhat(k - j), fhat(j - l), or fhat (l),
        # respectively
        if flag == 0
            for j = 1:m_size
                # colors all invalid elements of matrix red
                matches = 0
                for x = 1:3
                    if abs(m[i, j][x]) == n
                        # if m[i, j][x] == n
                        # if -m[i, j][x] == n
                        matches += 1
                    end
                end
                # if i > 2N + 2 - j
                #     print("   ")
                if matches != 0
                    matches_str = AnnotatedString(string(matches), [(1:9, :face, :green)])
                    print(matches_str, "  ")
                elseif j - N - 1 < -N + i - N - 1 + k || j - N - 1 > N + i - N - 1 + k
                    matches_str = AnnotatedString(string(matches), [(1:9, :face, :red)])
                    print(matches_str, "  ")
                else
                    print(matches, "  ")
                end
            end
        else
            for j = 1:m_size
                element_str = AnnotatedString(string(m[i, j][flag]), [(1:9, :face, :red)])
                if j - N - 1 < -N + i - N - 1 + k || j - N - 1 > N + i - N - 1 + k
                    print(element_str, " ")
                else
                    print(m[i, j][flag], " ")
                end
                if m[i, j][flag] >= 0 && j < 2N + 1 && m[i, j+1][flag] >= 0
                    print(" ")
                end
            end
        end
        println()
        println("    |")
    end
end
k = 1
N = 10
flag = 0

# print_eq(k, 15, 0, 4)
# print_eq(k, N, flag, 1)
# print_eq(k, N, flag, 2)
# print_eq(k, N, flag, 3)
# print_eq(k, N, flag, 4)

for k = 1:N
    println("----------------------------------------------")
    for n = 1:N
        print_eq(k, N, 0, n)
    end
end

