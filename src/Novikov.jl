module Novikov

include("tw.jl")
include("evolve.jl")
include("utils.jl")
include("fs.jl")

export
# modules
    Utils,

# functions
    evolve,
    construct_twsol,
    construct_jacobian!,
    F!,
    print_jac,
    writeoutput,
    readoutput,
    appendoutput,

# errors
    ConvergenceError,
    InsufficientModeError

end # module Novikov
