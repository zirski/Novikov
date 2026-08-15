using ProgressBars, FFTW

function extend_amp(L_start, L_end, idx_src)
    inpath = joinpath("output", "amp_range_" * Base.string(Int(L_start / pi)) * "pi.txt")
    outpath = joinpath("output", "L_ext_" * Base.string(Int(L_start / pi)) * "pi_" * Base.string(Int(L_end / pi)) * "pi.txt")

    in = sortoutput(inpath)
    num_dest = 1000
    N = in.header.N
    clist = in.fields[1]
    sols_src = in.fields[2]

    sol_src = vec(sols_src[idx_src, :])
    c = clist[idx_src]

    Lrange = collect(range(L_start, L_end, num_dest))
    sols_dest = Vector{Vector{Float64}}(undef, 0)

    push!(sols_dest, sol_src)
    prob = construct_twsol(rfft(sol_src) / N, c, Lrange[2], N_gp=N)
    push!(sols_dest, prob.sol)

    iter = tqdm(Lrange[3:end])
    for L in iter
        try
            prob = construct_twsol(prob.sol_hat, c, L, N_gp=N)
            push!(sols_dest, prob.sol)
            set_postfix(iter, Lines=L)
        catch e
            if e isa ConvergenceError
                throw(ArgumentError("Specified source solution cannot be extended; try one with a smaller amplitude"))
            else
                error(e)
            end
        end
    end

    header = FileHeader(c, L_start, N)
    writeoutput(outpath, header, (Lrange, sols_dest))
    log("Test completed; wrote " * string(num_dest) * " lines to " * string(outpath))
    return nothing
end

function amplim(inpath, L; dc=1 / 4096, max_q=2000, maxmodes=1024)
    outpath = joinpath("output", "amp_range_" * Base.string(Int(L / pi)) * "pi.txt")
    seed = getsol(inpath)
    seed_sol = seed.sol

    N = seed.N
    c = seed.c

    header = FileHeader(c, L, seed.N)
    writeoutput(outpath, header, ([c], seed_sol'))

    lines_written = 1

    sols = Vector{Vector{Float64}}(undef, 0)
    cs = Vector{Float64}(undef, 0)

    c_inc = c + dc
    c_dec = c - dc

    prob_inc = construct_twsol(rfft(seed_sol) / N, c_inc, L, N_gp=N, maxmodes=maxmodes)
    prob_dec = construct_twsol(rfft(seed_sol) / N, c_dec, L, N_gp=N, maxmodes=maxmodes)

    inc_lim = false
    dec_lim = false

    iter = tqdm(2:max_q)
    for i in iter
        if !inc_lim
            try
                prob_inc = construct_twsol(prob_inc.sol_hat, c_inc, L, N_gp=N, maxmodes=maxmodes)

                push!(sols, prob_inc.sol)
                push!(cs, c_inc)
                c_inc += dc
                lines_written += 1
            catch e
                if e isa InsufficientModeError
                    log("upper limit reached for " * string(N) * " modes. ")
                    inc_lim = true
                else
                    error(e)
                end
            end
        end

        if !dec_lim
            try
                prob_dec = construct_twsol(prob_dec.sol_hat, c_dec, L, N_gp=N, maxmodes=maxmodes)

                if isapprox(prob_dec.sol_hat[2], 0, atol=1e-15)
                    dec_lim = true
                    log("lower limit reached; i = " * string(i))
                else
                    pushfirst!(sols, prob_dec.sol)
                    pushfirst!(cs, c_dec)
                    c_dec -= dc
                    lines_written += 1
                end
            catch e
                if e isa ConvergenceError
                    dec_lim = true
                    log("lower limit reached; i = " * string(i))
                else
                    error(e)
                end
            end
        end

        # write a chunk of solutions to output
        if i % 10 == 0
            appendoutput(outpath, (cs, sols))
            # println(lines_written, " lines written to ", outpath, ".")
            empty!(sols)
            empty!(cs)
        end
        inc_lim && dec_lim && break
        set_postfix(iter, Lines=lines_written)
    end
    return nothing
end

function amplim(L; dc, max_q, maxmodes)
    inpath = joinpath("output", "L_ext_" * Base.string(Int(L / pi) - 1) * "pi_" * Base.string(Int(L / pi)) * "pi.txt")
    amplim(inpath, L, dc=dc, max_q=max_q, maxmodes=maxmodes)
    return nothing
end

function Plots.plot(sol::NovikovSolution; kwargs...)
    xvec = collect(0:(sol.N-1)) * sol.L / sol.N
    Plots.plot(xvec, sol.sol; kwargs...)
end

function Plots.plot!(sol::NovikovSolution; kwargs...)
    @nospecialize
    xvec = collect(0:(sol.N-1)) * sol.L / sol.N
    local plt
    try
        plt = Plots.current()
    catch
        return Plots.plot(sol; kwargs...)
    end
    return Plots.plot!(current(), xvec, sol.sol; kwargs...)
end
