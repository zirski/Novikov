module Novikov

include("tw.jl")
include("evolve.jl")

export gen_kvec, evolve, dscrt, integrate, deriv!, deriv, gen_tw_sol_2,
    gen_tw_sol_nl, gen_tw_sol_nonlinear, gen_tw_sol_3, gen_jacobian_1, gen_jacobian_2, gen_f_from_fhats, f_fourier, F, print_jac

end # module Novikov
