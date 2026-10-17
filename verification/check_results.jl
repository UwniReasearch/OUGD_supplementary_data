include(joinpath(@__DIR__, "..", "experiments", "common.jl"))

data = joinpath(ROOT, "data")
random = read_csv(joinpath(data, "random_initializations.csv"))
@assert length(random) == 100
@assert all(r -> r.num_bias_params == 1, random)
@assert all(r -> r.gradient_tolerance_met && r.stop_reason == "gradient_tol" &&
    r.max_iter == "none" && r.max_seconds == Inf && r.final_gradient_norm < 1e-8, random)
@assert [r.seed for r in random] == [i*1000+(i*7%999)+1 for i in 1:100]
history = read_csv(joinpath(data, "random_convergence_history.csv"))
for row in random
    trajectory = filter(h -> h.seed == row.seed, history)
    @assert length(trajectory) == row.iterations+1
    @assert [h.iteration for h in trajectory] == collect(0:row.iterations)
    @assert isapprox(last(trajectory).frobenius_cost,row.frobenius_cost; atol=1e-13)
end
scaling = read_csv(joinpath(data, "extended_scalability.csv"))
@assert [r.bits for r in scaling] == collect(2:9)
@assert all(r -> r.num_bias_params == 1, scaling)
@assert all(r -> r.gradient_tolerance_met && r.max_iter == "none" &&
    r.max_seconds == Inf && r.final_gradient_norm < 1e-8, scaling)
methods = read_csv(joinpath(data, "method_comparison.csv"))
@assert length(methods) == 72
@assert all(r -> r.num_bias_params == 1, methods)
@assert all(r -> r.gradient_tolerance_met && r.max_iter == "none" &&
    r.max_seconds == Inf && r.compile_seconds == 0 && r.recompile_seconds == 0, methods)
starts = read_csv(joinpath(data, "method_initializations.csv"))
@assert length(starts) == 30 && all(r -> r.gradient_tolerance_met, starts)
@assert all(r -> r.num_bias_params == 1, starts)
objectives = read_csv(joinpath(data, "objective_comparison_runs.csv"))
paired = read_csv(joinpath(data, "objective_comparison_pairs.csv"))
@assert length(objectives) == 2length(paired)
@assert length(paired) == 5 && [r.bits for r in paired] == collect(2:6)
@assert all(r -> r.gradient_backend == "analytic" && r.num_bias_params == 1 && r.final_gradient_norm < 1e-8, objectives)
@assert all(r -> r.max_iter == "none" && r.max_seconds == Inf &&
    r.requested_gtol == 1e-8 && r.stop_reason == "gradient_tol", objectives)
@assert all(r -> r.initial_U_max_difference == 0 && r.initial_bias_max_difference == 0,
    paired)
println("Single-bias tables: 100 initializations, complete trajectories, 8 sizes, 72 method timings, 30 paired starts and ",length(objectives)," objective runs checked.")

include(joinpath(@__DIR__, "check_interconnect_eight_layer.jl"))
