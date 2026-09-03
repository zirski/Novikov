function writesols(filename, sols::Vector{NovikovSolution}, ow = false)
    path = joinpath(dirname(Base.active_project()), filename)
    ow && rm(path, force = true)
    for sol in sols
        writesol(filename, sol)
    end
    return nothing
end

function writesol(filename, sol::NovikovSolution)
    H = maximum(sol.sol) - minimum(sol.sol)
    path = joinpath(dirname(Base.active_project()), filename)
    open(path, "a") do io
        data = hcat([sol.c, sol.L, H], reshape(sol.sol, 1, sol.N))
        write(io, join(data, '\t') * '\n')
    end
    return nothing
end

function readsols(filename)
    path = joinpath(dirname(Base.active_project()), filename)
    open(path, "r") do io
        sols = Vector{NovikovSolution}(undef, countlines(io))

        for (i, line) in enumerate(eachline(io))
            elements = parse.(Float64, split(line, '\t'))
            sols[i] = NovikovSolution(
                elements[1],
                elements[2],
                elements[3],
                elements[4:end],
                length(elements) - 3,
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
