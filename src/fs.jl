struct SolutionList
    header
    fields
    N
end

function writeoutput(name, header, fields::Vector{AbstractArray})
    path = joinpath(dirname(Base.active_project()), "output", string(name, ".txt"))
    rm(path, force=true)
    open(path, "a") do io
        write(io, header * '\n')
        data = reduce(hcat, fields)
        for row in eachrow(data)
            write(io, join(row, '\t') * '\n')
        end
    end
end

writeoutput(name, header, fields::Vector{Vector}) = writeoutput(name, header, permutedims.(fields))

function appendoutput(name, fields)
    path = joinpath(dirname(Base.active_project()), "output", string(name, ".txt"))
    open(path, "a") do io
        data = reduce(hcat, fields)
        for row in eachrow(data)
            write(io, join(row, '\t') * '\n')
        end
    end
end

function readoutput(name)
    path = joinpath(dirname(Base.active_project()), "output", string(name, ".txt"))
    xfield::Vector{Float64} = []
    yfield = []
    header = []
    open(path, "r") do io
        header = readline(io)
        for (i, line) in enumerate(eachline(io))
            line_elements = parse.(Float64, split(line, '\t'))
            push!(xfield, popfirst!(line_elements))
            yfield = i == 1 ? line_elements' : vcat(yfield, line_elements')
        end
    end
    p = sortperm(xfield)
    xfield .= xfield[p]
    yfield .= yfield[p, :]
    return SolutionList(header, (xfield, yfield), size(yfield, 2))
end
