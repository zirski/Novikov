struct FileHeader
    c
    L
    N
end

Base.string(h::FileHeader) = join((Base.string(h.c), Base.string(h.L), Base.string(h.N)), '\t')

function FileHeader(s::String)
    fields = parse.(Float64, split(s, '\t'))
    return FileHeader(fields[1], fields[2], Int(fields[3]))
end

struct SolutionList
    header::FileHeader
    fields
    N
end

function writeoutput(rpath, header::FileHeader, fields::Tuple{Vararg{AbstractVecOrMat{T}}}) where T<:Number
    rpath = joinpath(dirname(Base.active_project()), rpath)
    rm(rpath, force=true)
    open(rpath, "a") do io
        write(io, string(header) * '\n')
        data = reduce(hcat, fields)
        for row in eachrow(data)
            write(io, join(row, '\t') * '\n')
        end
    end
    return nothing
end

# TODO: Currently inputting a tuple of Vector{<:Number} will cause a stack overflow, I have no clue how to do the type
# stuff to fix. In the meantime, writes consisting of single solutions are not allowed unless the caller transposes the solution
# field so the other method is triggered.
function writeoutput(rpath, header::FileHeader, fields::Tuple{Vararg{Vector}})
    writeoutput(rpath, header, map(fields) do field
        if field isa Vector{Vector{T}} where T<:Number
            return reduce(vcat, permutedims.(field))
        else
            return field
        end
    end
    )
end

function writeoutput(sol::NovikovSolution)
    header = FileHeader(sol.c, sol.L, sol.N)
    writeoutput(joinpath("output", "seed_" * string(Int(sol.L / pi)) * "pi.txt"), header, ([sol.c], sol.sol'))
end
    
function appendoutput(rpath, fields::Tuple{Vararg{AbstractVecOrMat{T}}}) where T<:Number
    apath = joinpath(dirname(Base.active_project()), rpath)
    open(apath, "a") do io
        data = reduce(hcat, fields)
        for row in eachrow(data)
            write(io, join(row, '\t') * '\n')
        end
    end
    return nothing
end

function appendoutput(rpath, fields::Tuple{Vararg{Vector}})
    appendoutput(rpath, map(fields) do field
        if field isa Vector{Vector{T}} where T<:Number
            return reduce(vcat, permutedims.(field))
        else
            return field
        end
    end
    )
end
# 'rpath' refers to the path relative to the workspace root, which SHOULD work since julia should always be run from there.
function readoutput(rpath)
    apath = joinpath(dirname(Base.active_project()), rpath)
    open(apath, "r") do io
        header = FileHeader(readline(io))
        mark(io)
        num_rows = countlines(io)
        reset(io)

        xfield = zeros(num_rows)
        yfield = zeros(num_rows, header.N)

        for (i, line) in enumerate(eachline(io))
            line_elements = parse.(Float64, split(line, '\t'))
            xfield[i] = line_elements[1]
            yfield[i, :] .= line_elements[2:end]
        end
        # returns 2 vectors if only one line, 1 vector + 1 matrix if multiple
        if size(yfield, 1) == 1
            return SolutionList(header, (xfield, vec(yfield)), length(yfield))
        else
            return SolutionList(header, (xfield, yfield), size(yfield, 2))
        end
    end
end

function sortoutput(rpath; rev=false)
    ds = readoutput(rpath)
    xfield, yfield = ds.fields

    # dataset has only 1 row; already sorted
    if length(xfield) == 1
        return SolutionList(ds.header, (xfield, vec(yfield)), ds.N)
    end

    p = sortperm(xfield, rev=rev)
    xfield .= xfield[p]
    yfield .= yfield[p, :]

    if size(yfield, 1) == 1
        return SolutionList(ds.header, (xfield, vec(yfield)), ds.N)
    end
    return SolutionList(ds.header, (xfield, yfield), ds.N)
end

# TODO: current impl only works for files with fields like (c, sol); no current way to determine values of fields.
function getsol(rpath, by, val)
    ds = sortoutput(rpath)
    cs, sols = ds.fields
    heights = [maximum(sol) - minimum(sol) for sol in eachrow(sols)]
    tol = 1e-4

    queryfield = by == :c ? cs : heights

    idx = findall(x -> isapprox(x, val, atol=tol), queryfield)

    while (length(idx) == 0 || length(idx) > 1)
        if length(idx) == 0
            tol *= 10
            tol > 0.5 && throw(ArgumentError("No matches found."))
            idx = findall(x -> isapprox(x, val, atol=tol), queryfield)
        else
            tol /= 2
            idx = findall(x -> isapprox(x, val, atol=tol), queryfield)
        end
    end
    idx = idx[1]
    return NovikovSolution(cs[idx], ds.header.L, vec(sols[idx, :]), ds.header.N)
end

function getsol(rpath)
    ds = readoutput(rpath)
    return NovikovSolution(ds.header.c, ds.header.L, ds.fields[2], ds.header.N)
end

function getsol(rpath, idx)
    ds = sortoutput(rpath)
    f1, f2 = ds.fields
    qf1 = qf2 = 0.0

    if idx isa Number
        qf1 = f1[idx]
        qf2 = vec(f2[idx, :])
    else
        qf1 = f1[end]
        qf2 = vec(f2[end, :])
    end

    if splitpath(rpath)[end][1] == 'a'
        return NovikovSolution(qf1, ds.header.L, qf2, ds.header.N)
    else
        return NovikovSolution(ds.header.c, qf1, qf2, ds.header.N)
    end

end

