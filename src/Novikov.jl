module Novikov

include("tw.jl")
include("evolve.jl")
include("utils.jl")
include("fs.jl")
include("da.jl")
include("sta.jl")

const LOG_PATH = "/Users/tobyhammond/research/NovikovSolver/Novikov/log/log.txt"

export
    # functions
    dscrt,
    integrate,
    deriv!,
    deriv,
    evolve,
    construct_twsol,
    construct_jacobian!,
    F!,
    print_jac,
    writeoutput,
    readoutput,
    appendoutput,
    sortoutput,
    getsol,
    extend_amp,
    amplim,
    gen_kvec,
    istw,
    compute_evals,

    # errors
    ConvergenceError,
    InsufficientModeError,

    # structs
    NovikovProblem,
    NovikovSolution,
    FileHeader

end # module Novikov
