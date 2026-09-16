using UUIDs
using Printf
using FFTW: LinearAlgebra

function writesols(filename, sols::Vector{NovikovSolution}, ow = false)
    path = joinpath(dirname(Base.active_project()), filename)
    ow && rm(path, force = true)
    for sol in sols
        writesol(filename, sol)
    end
    return nothing
end

function writesol(filename, sol::NovikovSolution)
    path = joinpath(dirname(Base.active_project()), filename)
    open(path, "a") do io
        data =
            hcat(reshape([sol.c, sol.L, sol.H], 1, 3), reshape(sol.sol, 1, sol.N))
        write(io, "$(sol.id)\t" * join(data, '\t') * '\n')
    end
    return nothing
end

function readsols(filename)
    path = joinpath(dirname(Base.active_project()), filename)
    open(path, "r") do io
        sols = Vector{NovikovSolution}(undef, countlines(io))

        seekstart(io)

        for (i, line) in enumerate(eachline(io))
            elements = split(line, '\t')
            sols[i] = NovikovSolution(
                UUID(elements[1]),
                parse(Float64, elements[2]),
                parse(Float64, elements[3]),
                parse(Float64, elements[4]),
                parse.(Float64, elements[5:end]),
                length(elements[5:end]),
            )
        end
        return sols
    end
end

function sortsols(filename, ow::Bool)
    sols = readsols(filename)
    sort!(sols)
    ow && writesols(filename, sols, ow)
    return sols
end

function getsol(filename, by::Symbol, val)
    by == :id && throw(ArgumentError("Can't search by id; are you insane?"))
    tol = 1e-4
    sols = sortsols(filename, false)
    idx = findall(x -> isapprox(getfield(x, by), val, atol = tol), sols)

    while (length(idx) == 0 || length(idx) > 1)
        if length(idx) == 0
            tol *= 10
            tol > 0.5 && throw(ArgumentError("No matches found."))
            idx = findall(x -> isapprox(getfield(x, by), val, atol = tol), sols)
        else
            tol /= 2
            idx = findall(x -> isapprox(getfield(x, by), val, atol = tol), sols)
        end
        tol < 1e-10 &&
            throw(ArgumentError("Specified field has too many matches."))
    end
    return sols[only(idx)]
end

function getsol(filename)
    try
        return only(readsols(filename))
    catch e
        if e isa ArgumentError
            throw(ArgumentError("Specified file has more than one solution."))
        else
            error(e)
        end
    end
end

getsol(filename, idx) = sortsols(filename)[idx == :end ? end : idx]
