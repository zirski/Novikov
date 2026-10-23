using ProgressBars
using FFTW
using Plots
using Printf
using LaTeXStrings
using UUIDs


"""
    extendpd(L_start, L_end, sol_src; num_sols=1000, c_end=sol_src.c)

Continue the traveling wave `sol_src` in period from `L_start` to `L_end`,
computing `num_sols` solutions (including `sol_src`), and write them to
`output/pextn/<L_start>_<L_end>.txt` (periods to 4 decimal places), replacing
any existing file.

The wave speed moves linearly from `sol_src.c` to `c_end` alongside the period.
Shrinking the period at fixed speed makes the waves increasingly peaked, so
lower `c_end` to keep them smooth; the default keeps the speed fixed.
"""
function extendpd(L_start, L_end, sol_src; num_sols=1000, c_end=sol_src.c)
    log("Starting period extension---------------------------------")

    outpath = @sprintf "output/pextn/%.4f_%.4f.txt" L_start L_end

    Lrange = range(L_start, L_end, num_sols)
    crange = range(sol_src.c, c_end, num_sols)
    sols = Vector{NovikovSolution}(undef, 0)
    push!(sols, sol_src)

    # Fourier coefficients of the last solution found, used as the initial
    # guess for the next one
    guess = rfft(sol_src.sol) / sol_src.N

    iter = tqdm(zip(Lrange[2:end], crange[2:end]))
    for (i, (L, c)) in enumerate(iter)
        try
            prob = construct_twsol(guess, c, L, N_gp=sol_src.N)
            guess = prob.sol_hat
            push!(sols, NovikovSolution(prob))
            set_postfix(iter, Lines=i)
        catch e
            if e isa ConvergenceError
                throw(
                    ArgumentError(
                        "Specified source solution cannot be extended; try one with a smaller amplitude or a lower c_end",
                    ),
                )
            else
                error(e)
            end
        end
    end

    writesols(outpath, sols, true)

    log(
        "Test completed; wrote " *
            string(num_sols) *
            " lines to " *
            string(outpath),
    )
    return nothing
end

"""
    amplim(inpath, L; dc=1/4096, max_q=2000, maxmodes=1024)

Find the range of wave speeds (and so amplitudes) of traveling waves with
period `L` by continuing a seed solution from `inpath` in steps of `dc` in both
directions, for at most `max_q` steps. Continuation upward stops when more than
`maxmodes` modes are needed, and downward when the solution becomes trivial or
Newton's method fails. Solutions are written to `output/ampranges/<L>.txt`
(`L` to 4 decimal places).
"""
function amplim(inpath, L; dc=1 / 4096, max_q=2000, maxmodes=1024)
    log("Starting amplitude rangefinding test------------------------- ")

    outpath = @sprintf "output/ampranges/%.4f.txt" L

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

    sols = Vector{NovikovSolution}(undef, 0)
    push!(sols, seedsol)

    c_inc = seedsol.c + dc
    c_dec = seedsol.c - dc

    # Fourier coefficients of the last solution found in each direction, used
    # as the initial guess for the next one
    guess_inc = guess_dec = rfft(seedsol.sol) / seedsol.N

    inc_lim = false
    dec_lim = false

    iter = tqdm(2:max_q)
    for i in iter
        if !inc_lim
            try
                prob_inc = construct_twsol(
                    guess_inc,
                    c_inc,
                    L,
                    N_gp=seedsol.N,
                    maxmodes=maxmodes,
                )

                guess_inc = prob_inc.sol_hat
                push!(sols, NovikovSolution(prob_inc))
                c_inc += dc
                lines_written += 1
            catch e
                if e isa InsufficientModeError
                    log(
                        "upper limit reached; more than " *
                            string(maxmodes) *
                            " modes needed. ",
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
                    guess_dec,
                    c_dec,
                    L,
                    N_gp=seedsol.N,
                    maxmodes=maxmodes,
                )

                if isapprox(prob_dec.sol_hat[2], 0, atol=1e-15)
                    dec_lim = true
                    log("lower limit reached; i = " * string(i))
                else
                    guess_dec = prob_dec.sol_hat
                    pushfirst!(sols, NovikovSolution(prob_dec))
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
        set_postfix(iter, Lines=lines_written)
    end
    # write whatever was found since the last chunk
    writesols(outpath, sols, true)
    log(
        "Test completed; wrote " *
            string(lines_written) *
            " lines to " *
            string(outpath),
    )
    return nothing
end

"""
    plot(sol::NovikovSolution; kwargs...)

Plot `sol` over one period.
"""
function Plots.plot(sol::NovikovSolution; kwargs...)
    xvec = collect(0:(sol.N-1)) * sol.L / sol.N
    Plots.plot(xvec, sol.sol; kwargs...)
end

"""
    plot!(sol::NovikovSolution; kwargs...)

Add `sol` to the current plot, or create a new plot if there is none.
"""
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

"""
    plotevals(in, res, out)

Scatter-plot every `res`th stored eigenvalue of the solution in `in` in the
complex plane and save the figure to the `plots` directory.
"""
function plotevals(in, res, out)
    evals = getevals(in)

    Plots.plot(
        real(evals[1:res:end]),
        imag(evals[1:res:end]),
        seriestype=:scatter,
        label="",
        fontfamily="Computer Modern",
        xlabel=L"\Re~(\lambda)",
        ylabel=L"\Im~(\lambda)",
    )
    Plots.savefig(joinpath(dirname(Base.active_project()), "plots", pname))
    return nothing
end

function plotevals(sol)
    evals = getevals(sol)
    display(
        Plots.plot(
            real(evals),
            imag(evals),
            seriestype=:scatter,
            label="",
            ms=1
        )
    )
    return nothing

end
