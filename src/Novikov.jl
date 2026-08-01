module Novikov

include("tw.jl")
include("evolve.jl")

export gen_kvec,
    evolve,
    dscrt,
    integrate,
    deriv!,
    deriv,
    construct_twsol,
    construct_jacobian!,
    F!,
    print_jac,
    ConvergenceError,
    InsufficientModeError

end # module Novikov
