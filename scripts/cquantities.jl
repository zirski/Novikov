using Novikov, Revise, Elliptic, Elliptic.Jacobi, Random, FFTW, Plots, Printf

trig(x, a, b) = a * cos(b * pi * x) + a
jacobi(x, a, b, m) = a * (cn(b * x, m))^2
peak(x, c) = sqrt(c) * exp(-abs(x - 50))

num_tests = 10
N = 1024 * 4
t_f = 1.0
q = 1000 * 10
amplitude_max = 0.1

# trig parameters
cos_amps = rand(num_tests) * amplitude_max
cos_scale = rand(num_tests) * 5
Ls_trig = 2 ./ cos_scale
trig_kvecs = gen_kvec.(Ls_trig, N)

# cn parameters
cn_amps = rand(num_tests) * amplitude_max
cn_params = rand(num_tests) .* (1.0 - 2 * eps()) .+ eps()
b_cn = @. sqrt(cn_amps / (12 * cn_params))
Ls_cn = @. 2 * K(cn_params) / b_cn
cn_kvecs = gen_kvec.(Ls_cn, N)

# peak parameters
peak_cs = rand(num_tests) .* (1.0 - 2 * eps()) .+ eps()
L_peak = 100.0
peak_kvec = gen_kvec(L_peak, N)

# fourier setup
u_dummy = zeros(Float64, N)
uhat = Vector{ComplexF64}(undef, div(N, 2) + 1)
plan = plan_rfft(u_dummy)
iplan = plan_irfft(uhat, N)

mutable struct TestCase
        xvec::Vector{Float64}
        u_initial::Vector{Float64}
        u_final::Vector{Float64}
        id::String
        kvec::Vector{ComplexF64}
        params::Tuple
        results::Vector{Bool}
        errors::Vector{Float64}
end

function gen_test_case(id, params, N)
        xvec = zeros(Float64, N)
        init_profile = zeros(Float64, N)
        apx_profile = zeros(Float64, N)
        kvec = zeros(Float64, div(N, 2))
        if id == "trig"
                kvec = gen_kvec(params[3], N)
                xvec, init_profile = dscrt(x -> trig(x, params[1], params[2]), params[3], N)
                apx_profile = evolve(init_profile, t_f, q, kvec, N)
                return TestCase(xvec, init_profile, apx_profile, id, kvec, params, Vector{Bool}(undef, 3), zeros(Float64, 3))
        elseif id == "cn"
                kvec = gen_kvec(params[4], N)
                xvec, init_profile = dscrt(x -> jacobi(x, params[1], params[2], params[3]), params[4], N)
                apx_profile = evolve(init_profile, t_f, q, kvec, N)
                return TestCase(xvec, init_profile, apx_profile, id, kvec, params, Vector{Bool}(undef, 3), zeros(Float64, 3))
        elseif id == "peak"
                kvec = gen_kvec(params[2], N)
                xvec, init_profile = dscrt(x -> peak(x, params[1]), params[2], N)
                apx_profile = evolve(init_profile, t_f, q, kvec, N)
                return TestCase(xvec, init_profile, apx_profile, id, kvec, params, Vector{Bool}(undef, 3), zeros(Float64, 3))
        end
end

function cq_test(expr, tc::TestCase, tol, cq_num)
        L = 0
        if tc.id == "trig"
                L = tc.params[3]
        elseif tc.id == "cn"
                L = tc.params[4]
        elseif tc.id == "peak"
                L = tc.params[2]
        end

        cq_initial = integrate(expr(tc.u_initial, tc.kvec), L, N)
        cq_final = integrate(expr(tc.u_final, tc.kvec), L, N)
        tc.results[cq_num] = isapprox(cq_initial, cq_final, atol=tol)
        tc.errors[cq_num] = abs(cq_initial - cq_final)
end

function cq1_calc(u, kvec)
        u_x = deriv(u, uhat, kvec, plan, iplan)
        return @. 1 / 8 * (u^4 + 2 * u^2 * u_x^2 - u_x^4 / 3)
end

function cq2_calc(u, kvec)
        return cbrt.(u .- deriv(u, uhat, 2, kvec, plan, iplan)) .^ 2
end

function cq3_calc(u, kvec)
        m = u - deriv(u, uhat, 2, kvec, plan, iplan)
        m_x = deriv(m, uhat, kvec, plan, iplan)
        return @. 1 / 3 * (cbrt(m)^-8 * m_x^2 + 9 * cbrt(m)^-2)
end

tol = 1e-10
test_cases = Array{TestCase}(undef, num_tests, 3)
for i = 1:num_tests
        test_cases[i, 1] = gen_test_case("trig", (cos_amps[i], cos_scale[i], Ls_trig[i]), N)
        test_cases[i, 2] = gen_test_case("cn", (cn_amps[i], b_cn[i], cn_params[i], Ls_cn[i]), N)
        test_cases[i, 3] = gen_test_case("peak", (peak_cs[i], L_peak), N)

        for j = 1:3
                cq_test(cq1_calc, test_cases[i, j], tol, 1)
                cq_test(cq2_calc, test_cases[i, j], tol, 2)
                cq_test(cq3_calc, test_cases[i, j], tol, 3)
        end
end

function plot_tc(tc::TestCase)
        plot(tc.xvec, tc.u_initial)
        display(plot!(tc.xvec, tc.u_final))
end

print_sci(num) = @sprintf("%4.8e", num)
print_reg(num) = @sprintf("%010.8f", num)

println("--------------------------------Test results------------------------------------------")
println("------------CQ1-----------")
println("Trig functions")
println("Test no.\t|Pass/fail\t|Quantity difference\t|Amplitude\t|Scale\t\t|Period")
println()
for i = 1:num_tests
        local tc::TestCase = test_cases[i, 1]
        println(
                i,
                "\t\t|",
                tc.results[1] ? "Pass" : "Fail",
                "\t\t|",
                print_sci(tc.errors[1]),
                "\t\t|",
                print_reg(tc.params[1]),
                "\t|",
                print_reg(tc.params[2]),
                "\t|",
                print_reg(tc.params[3])
        )
end

println("Cnoidal waves")
println("Test no.\t|Pass/fail\t|Quantity difference\t|Amplitude\t|Scale\t\t|Parameter\t|Period")
println()
for i = 1:num_tests
        local tc::TestCase = test_cases[i, 2]
        println(
                i,
                "\t\t|",
                tc.results[1] ? "Pass" : "Fail",
                "\t\t|",
                print_sci(tc.errors[1]),
                "\t\t|",
                print_reg(tc.params[1]),
                "\t|",
                print_reg(tc.params[2]),
                "\t|",
                print_reg(tc.params[3]),
                "\t|",
                print_reg(tc.params[4])
        )
end

println("Peakons")
println("Test no.\t|Pass/fail\t|Quantity difference\t|speed (c)")
println()
for i = 1:num_tests
        local tc::TestCase = test_cases[i, 3]
        println(
                i,
                "\t\t|",
                tc.results[1] ? "Pass" : "Fail",
                "\t\t|",
                print_sci(tc.errors[1]),
                "\t\t|",
                print_reg(tc.params[1]),
        )
end
println("----------------------------------------------------------------------------------------------------------",
        "------------------------------------------")
println("------------CQ2-----------")
println("Trig functions")
println("Test no.\t|Pass/fail\t|Quantity difference\t|Amplitude\t|Scale\t\t|Period")
println("----------------------------------------------------------------------------------------------------------",
        "------------------------------------------")
for i = 1:num_tests
        local tc::TestCase = test_cases[i, 1]
        println(
                i,
                "\t\t|",
                tc.results[2] ? "Pass" : "Fail",
                "\t\t|",
                print_sci(tc.errors[2]),
                "\t\t|",
                print_reg(tc.params[1]),
                "\t|",
                print_reg(tc.params[2]),
                "\t|",
                print_reg(tc.params[3])
        )
end

println("Cnoidal waves")
println("Test no.\t|Pass/fail\t|Quantity difference\t|Amplitude\t|Scale\t\t|Parameter\t|Period")
println()
for i = 1:num_tests
        local tc::TestCase = test_cases[i, 2]
        println(
                i,
                "\t\t|",
                tc.results[2] ? "Pass" : "Fail",
                "\t\t|",
                print_sci(tc.errors[2]),
                "\t\t|",
                print_reg(tc.params[1]),
                "\t|",
                print_reg(tc.params[2]),
                "\t|",
                print_reg(tc.params[3]),
                "\t|",
                print_reg(tc.params[4])
        )
end

println("Peakons")
println("Test no.\t|Pass/fail\t|Quantity difference\t|speed (c)")
println()
for i = 1:num_tests
        local tc::TestCase = test_cases[i, 3]
        println(
                i,
                "\t\t|",
                tc.results[2] ? "Pass" : "Fail",
                "\t\t|",
                print_sci(tc.errors[2]),
                "\t\t|",
                print_reg(tc.params[1]),
        )
end
println("----------------------------------------------------------------------------------------------------------",
        "------------------------------------------")
println("------------CQ3-----------")
println("Trig functions")
println("Test no.\t|Pass/fail\t|Quantity difference\t|Amplitude\t|Scale\t\t|Period")
println()
for i = 1:num_tests
        local tc::TestCase = test_cases[i, 1]
        println(
                i,
                "\t\t|",
                tc.results[3] ? "Pass" : "Fail",
                "\t\t|",
                print_sci(tc.errors[3]),
                "\t\t|",
                print_reg(tc.params[1]),
                "\t|",
                print_reg(tc.params[2]),
                "\t|",
                print_reg(tc.params[3])
        )
end

println("Cnoidal waves")
println("Test no.\t|Pass/fail\t|Quantity difference\t|Amplitude\t|Scale\t\t|Parameter\t|Period")
println()
for i = 1:num_tests
        local tc::TestCase = test_cases[i, 2]
        println(
                i,
                "\t\t|",
                tc.results[3] ? "Pass" : "Fail",
                "\t\t|",
                print_sci(tc.errors[3]),
                "\t\t|",
                print_reg(tc.params[1]),
                "\t|",
                print_reg(tc.params[2]),
                "\t|",
                print_reg(tc.params[3]),
                "\t|",
                print_reg(tc.params[4])
        )
end

println("Peakons")
println("Test no.\t|Pass/fail\t|Quantity difference\t|speed (c)")
println()
for i = 1:num_tests
        local tc::TestCase = test_cases[i, 3]
        println(
                i,
                "\t\t|",
                tc.results[3] ? "Pass" : "Fail",
                "\t\t|",
                print_sci(tc.errors[3]),
                "\t\t|",
                print_reg(tc.params[1]),
        )
end