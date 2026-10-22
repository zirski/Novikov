using UUIDs
using Printf
using FFTW: LinearAlgebra

"""
    writesols(filename, sols, ow=false)

Append each solution in `sols` to `filename` (relative to the active project
directory); see [`writesol`](@ref). Delete the file first if `ow` is `true`.
"""
function writesols(filename, sols::Vector{NovikovSolution}, ow = false)
    path = joinpath(dirname(Base.active_project()), filename)
    ow && rm(path, force = true)
    for sol in sols
        writesol(filename, sol)
    end
    return nothing
end

"""
    writesol(filename, sol)

Append `sol` to `filename` (relative to the active project directory) as one
tab-separated line: `id`, `c`, `L`, `H`, then the solution values.
"""
function writesol(filename, sol::NovikovSolution)
    path = joinpath(dirname(Base.active_project()), filename)
    open(path, "a") do io
        data =
            hcat(reshape([sol.c, sol.L, sol.H], 1, 3), reshape(sol.sol, 1, sol.N))
        write(io, "$(sol.id)\t" * join(data, '\t') * '\n')
    end
    return nothing
end

"""
    readsols(filename)

Read all solutions from `filename` (relative to the active project directory),
in the format written by [`writesol`](@ref), and return them as a
`Vector{NovikovSolution}`.
"""
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

"""
    sortsols(filename, ow=false)

Read the solutions in `filename` and return them sorted by wave speed. If `ow`
is `true`, overwrite the file with the sorted solutions.
"""
function sortsols(filename, ow::Bool=false)
    sols = readsols(filename)
    sort!(sols)
    ow && writesols(filename, sols, ow)
    return sols
end

"""
    getsol(filename, by::Symbol, val)

Return the unique solution in `filename` whose field `by` (e.g. `:c`, `:L`,
`:H`) is closest to `val`, adjusting the matching tolerance until exactly one
solution matches. Throw an `ArgumentError` if none or too many match.
"""
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

"""
    getsol(filename)

Return the solution in `filename`, which must contain exactly one.
"""
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

"""
    getsol(filename, idx)

Return the `idx`th solution in `filename` after sorting by wave speed; `idx`
may be `:end` for the last.
"""
getsol(filename, idx) = sortsols(filename, false)[idx == :end ? end : idx]

"""
    write_evals(filepath, sol, n_pts)

Compute the eigenvalues of the linearization about `sol` for `n_pts` Floquet
parameters in `[-π/L, π/L]` and write them to `filepath`: the solution's UUID on
the first line, then one eigenvalue per line as printed by Julia (`x + yim`).
"""
function write_evals(filepath, sol::NovikovSolution, n_pts)
    evals = []

    log("Initiating eigval computation")
    for mu in tqdm(range(-pi / sol.L, pi / sol.L, n_pts))
        eval = compute_evals(sol, mu)
        push!(evals, eval)
    end

    evals = reduce(vcat, evals)

    rm(filepath, force = true)
    open(filepath, "a") do file
        println(file, sol.id)
        for eval in evals
            println(file, eval)
        end
        println("Wrote ", length(evals), " eigenvalues to ", filepath, ".")
        return nothing
    end
end

"""
    write_eigen(filepath, sol, n_pts)

Like [`write_evals`](@ref), but also write each eigenvector, tab-separated after
its eigenvalue.
"""
function write_eigen(filepath, sol::NovikovSolution, n_pts)
    evals = []
    evecs = []

    log("Initiating eigval, eigvec computation")

    for mu in tqdm(range(-pi / sol.L, pi / sol.L, n_pts))
        eigen = compute_evals(sol, mu, evecs = true)
        push!(evals, eigen.values)
        push!(evecs, eigen.vectors)
    end

    evals = reduce(vcat, evals)
    evecs = permutedims(reduce(hcat, evecs))

    rm(filepath, force = true)
    open(filepath, "a") do file
        println(file, sol.id)
        for (eval, evec) in zip(evals, eachrow(evecs))
            println(file, eval, '\t', join(vec(evec), '\t'))
        end
        println("Wrote eigenvalues and eigenvectors to ", filepath, ".")
        return nothing
    end

end

"""
    write_evals(sol, n_pts)

Write the eigenvalues of `sol` to `output/evals/<c>_<L>.txt` in the active
project directory.
"""
function write_evals(sol::NovikovSolution, n_pts)
    filename = @sprintf "%.4f_%.4f.txt" sol.c sol.L
    filepath = joinpath(dirname(Base.active_project()), "output/evals", filename)

    write_evals(filepath, sol, n_pts)
    return nothing
end


"""
    getevals(filename)

Return the stored eigenvalues of the single solution in `filename`.
"""
function getevals(filename)
    return getevals(getsol(filename))
end

"""
    getevals(sol::NovikovSolution)

Return the stored eigenvalues of `sol`, found by its UUID among the files in
the eigenvalue output directory.
"""
function getevals(sol::NovikovSolution)
    eval_files = Dict{UUID,String}()
    dirpath = "/Users/tobyhammond/research/NovikovSolver/output/evals/"

    for filename in readdir(dirpath)
        eval_files[UUID(readline(joinpath(dirpath, filename)))] = filename
    end

    open(joinpath(dirpath, eval_files[sol.id]), "r") do file
        readline(file)
        # Julia prints non-finite imaginary parts as e.g. "NaN*im", which parse
        # can't read; "NaNim" parses fine
        return [parse(ComplexF64, replace(line, "*" => "")) for line in eachline(file)]
    end
end

"""
    geteigen(filename)

Read the eigenvalue-eigenvector pairs written by [`write_eigen`](@ref) from
`filename` and return them as a vector of `(eval, evec)` tuples.
"""
function geteigen(filename)
    open(filename, "r") do file
        readline(file)
        epairs = []

        for line in eachline(file)
            # see getevals for why "*" is stripped
            elements = parse.(ComplexF64, split(replace(line, "*" => ""), '\t'))
            push!(epairs, (popfirst!(elements), elements))
        end
        return epairs
    end
end
