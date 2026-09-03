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
     
    # utils
    dscrt,
    integrate,
    deriv!,
    deriv,
    evolve,

    # traveling waves
    construct_twsol,
    construct_jacobian!,
    F!,
    print_jac,

    # filesystem
    writesol,
    writesols,
    sortsols,
    getsol,

    # data analysis
    extend_amp,
    amplim,
    gen_kvec,
    istw,
    compute_evals,
    write_evals,
    plot_evals,

    # errors
    ConvergenceError,
    InsufficientModeError,

    # structs
    NovikovProblem,
    NovikovSolution,

end # module Novikov
