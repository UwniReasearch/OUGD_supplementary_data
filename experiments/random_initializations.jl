include("common.jl")
runs = parse(Int, option("--runs", "100"))
runs > 0 || error("runs must be positive.")
rows, histories = NamedTuple[], NamedTuple[]
warm = UnitaryDecoder(3, 1; cost_type=NORM, random_seed=42)
optimize!(warm; verbose=false, gradient_backend=:analytic,
    max_iter=nothing, max_seconds=Inf, gtol=1e-8, stats=OptimizationStats())
for i in 1:runs
    seed = i*1000 + (i*7 % 999) + 1
    solver = UnitaryDecoder(3, 1; cost_type=NORM, random_seed=seed)
    initial = initialize_point(solver)
    stats = OptimizationStats()
    result, history = D._optimize(solver, initial.U, initial.params;
        gradient_backend=:analytic, record=true, max_iter=nothing,
        max_seconds=Inf, gtol=1e-8, stats)
    analysis = analyze_result(solver, result)
    gradient = gradient_norm(solver, result; gradient_backend=:analytic)
    push!(rows, (; bits=3, ports=8, num_bias_params=solver.num_params, seed, gradient_backend="analytic",
        max_iter="none", max_seconds=Inf, requested_gtol=1e-8,
        iterations=stats.iterations, stop_reason=string(result.stop_reason),
        final_gradient_norm=gradient, gradient_tolerance_met=gradient<1e-8,
        frobenius_cost=analysis.cost, frobenius_cost_8_decimal=@sprintf("%.8f",analysis.cost),
        unitarity_residual=norm(result.U'*result.U-I)))
    initial_cost = solver.cost_fn(initial.U*solver.build_input(initial.params))
    push!(histories, (;seed, num_bias_params=solver.num_params, iteration=0, frobenius_cost=initial_cost))
    for (iteration, (_, cost)) in enumerate(history)
        push!(histories, (;seed, num_bias_params=solver.num_params, iteration, frobenius_cost=cost))
    end
    write_csv(joinpath(OUT, "random_initializations.csv"), rows)
    write_csv(joinpath(OUT, "random_convergence_history.csv"), histories)
end
println(count(r -> r.gradient_tolerance_met, rows), "/", runs, " met the gradient tolerance.")
