module Novikov

include("tw.jl")
include("evolve.jl")
include("utils.jl")
include("fs.jl")
include("da.jl")
include("sta.jl")

const LOG_PATH = "/Users/tobyhammond/research/NovikovSolver/Novikov/log/log.txt"

export
    # utils
    dscrt,
    integrate,
    deriv!,
    deriv,
    evolve,
    kvec,

    # traveling waves
    construct_twsol,
    construct_jacobian!,
    F!,
    print_jac,

    # filesystem
    readsols,
    writesol,
    writesols,
    sortsols,
    getsol,
    resize,

    # data analysis
    extendamp,
    amplim,
    istw,
    compute_evals,
    write_evals,
    getevals,
    plotevals,
    write_eigen,
    geteigen,

    # errors
    ConvergenceError,
    InsufficientModeError,

    # structs
    NovikovProblem,
    NovikovSolution

end # module Novikov
