using ProgressBars
using FFTW
using Plots
using Printf
using LaTeXStrings
using UUIDs


function extendamp(L_start, L_end, sol_src; num_sols = 1000)
    log("Starting period extension---------------------------------")

    Lrange = collect(range(L_start, L_end, num_sols))
    sols = Vector{NovikovSolution}(undef, 0)

    push!(sols, sol_src)
    prob = construct_twsol(
        rfft(sol_src) / sol_src.N,
        sol_src.c,
        Lrange[2],
        N_gp = sol_src.N,
    )
    sol = NovikovSolution(uuid4(), sol_src.c, L_start, prob.H, prob.sol, sol_src.N)
    push!(sols, sol)

    iter = tqdm(Lrange[3:end])
    for L in iter
        try
            prob = construct_twsol(prob.sol_hat, sol_src.c, L, N_gp = sol_src.N)
            sol = NovikovSolution(uuid4(), sol_src.c, L, prob.H, prob.sol, sol_src.N)
            push!(sols, prob.sol)
            set_postfix(iter, Lines = L)
        catch e
            if e isa ConvergenceError
                throw(
                    ArgumentError(
                        "Specified source solution cannot be extended; try one with a smaller amplitude",
                    ),
                )
            else
                error(e)
            end
        end
    end

    writesols("$(L_start)_($L_end).txt", sols)

    log(
        "Test completed; wrote " *
        string(num_sols) *
        " lines to " *
        string(outpath),
    )
    return nothing
end

function amplim(inpath, L; dc = 1 / 4096, max_q = 2000, maxmodes = 1024)
    log("Starting amplitude rangefinding test------------------------- ")

    outpath = "output/ampranges/" * string(L) * ".txt"

    local seedsol
    try
        seedsol = getsol(inpath)
    catch e
        if e isa ArgumentError
            seedsol = getsol(inpath, :L, L)
        else
            error(e)
        end
    end

    lines_written = 1

    sols = Vector{Vector{Float64}}(undef, 0)
    push!(sols, seedsol)
    cs = Vector{Float64}(undef, 0)

    c_inc = seedsol.c + dc
    c_dec = seedsol.c - dc

    prob_inc = construct_twsol(
        rfft(seedsol.sol) / seedsol.N,
        c_inc,
        L,
        N_gp = seedsol.N,
        maxmodes = maxmodes,
    )
    prob_dec = construct_twsol(
        rfft(seedsol.sol) / seedsol.N,
        c_dec,
        L,
        N_gp = seedsol.N,
        maxmodes = maxmodes,
    )

    inc_lim = false
    dec_lim = false

    iter = tqdm(2:max_q)
    for i in iter
        if !inc_lim
            try
                prob_inc = construct_twsol(
                    prob_inc.sol_hat,
                    c_inc,
                    L,
                    N_gp = seedsol.N,
                    maxmodes = maxmodes,
                )

                push!(sols, prob_inc.sol)
                push!(cs, c_inc)
                c_inc += dc
                lines_written += 1
            catch e
                if e isa InsufficientModeError
                    log(
                        "upper limit reached for " *
                        string(seedsol.N) *
                        " modes. ",
                    )
                    inc_lim = true
                else
                    error(e)
                end
            end
        end

        if !dec_lim
            try
                prob_dec = construct_twsol(
                    prob_dec.sol_hat,
                    c_dec,
                    L,
                    N_gp = seedsol.N,
                    maxmodes = maxmodes,
                )

                if isapprox(prob_dec.sol_hat[2], 0, atol = 1e-15)
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
            writesols(outpath, sols, true)
        end
        inc_lim && dec_lim && break
        set_postfix(iter, Lines = lines_written)
    end
    log(
        "Test completed; wrote " *
        string(lines_written) *
        " lines to " *
        string(outpath),
    )
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

function plotevals(in, res, out)
    evals = getevals(in)

    Plots.plot(
        real(evals[1:res:end]),
        imag(evals[1:res:end]),
        seriestype = :scatter,
        label = "",
        fontfamily = "Computer Modern",
        xlabel = L"\Re~(\lambda)",
        ylabel = L"\Im~(\lambda)",
    )
    Plots.savefig(joinpath(dirname(Base.active_project()), "plots", pname))
    return nothing
end
