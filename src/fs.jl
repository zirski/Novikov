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

getsol(filename, idx) = sortsols(filename, false)[idx == :end ? end : idx]

function write_evals(filepath, sol::NovikovSolution, n_pts)
    evals = []

    log("Initiating eigval computation")
    for mu in tqdm(range(-pi / sol.L, pi / sol.L, n_pts))
        eval = compute_evals(sol, mu)
        push!(evals, eval)
    end

    evals = reduce(vcat, evals)
    filter!(x -> abs(real(x)) > 1e-10, evals)

    rm(filepath, force = true)
    open(filepath, "a") do file
        println(file, sol.id)
        for eval in evals
            println(file, real(eval), '\t', imag(eval))
        end
        println("Wrote eigenvalues to ", filepath, ".")
        return nothing
    end
end

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
            println(
                file,
                real(eval),
                '\t',
                imag(eval),
                '\t',
                join(vec(evec), '\t'),
            )
        end
        println("Wrote eigenvalues and eigenvectors to ", filepath, ".")
        return nothing
    end

end

function write_evals(sol::NovikovSolution, n_pts)
    filename = @sprintf "%.4f_%.4f.txt" sol.c sol.L
    filepath = joinpath(Base.active_project(), "output/evals", filename)

    write_evals(filepath, sol, n_pts)
    return nothing
end


function getevals(filename)
    return getevals(getsol(filename))
end

function getevals(sol::NovikovSolution)
    eval_files = Dict{UUID,String}()
    dirpath = joinpath(Base.active_project(), "output/evals")

    for filename in readdir(dirpath)
        eval_files[UUID(readline(joinpath(dirpath, filename)))] = filename
    end

    open(eval_files[sol.id], "r") do file
        evals = Vector{ComplexF64}(undef, countlines(file))
        seekstart(file)
        readline(file)

        for (i, line) in enumerate(eachline(file))
            real, imag = parse.(Float64, split(line, '\t'))
            evals[i] = Complex(real, imag)
        end

        return evals
    end
end

function geteigen(filename)
    open(filename, "r") do file
        readline(file)
        epairs = []

        for line in eachline(file)
            elements = split(line, '\t')
            eval = Complex(
                parse(Float64, popfirst!(elements)),
                parse(Float64, popfirst!(elements)),
            )
            evec = parse.(ComplexF64, elements)
            push!(epairs, (eval, evec))
        end
        return epairs
    end
end
