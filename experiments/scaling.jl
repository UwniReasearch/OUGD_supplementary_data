include("common.jl")
max_bits = parse(Int, option("--max-bits", "9"))
2 <= max_bits <= 9 || error("Expected max-bits between 2 and 9.")

warm = UnitaryDecoder(2, 1; cost_type=NORM, random_seed=42)
warm_initial = initialize_point(warm)
D._optimize(warm, warm_initial.U, warm_initial.params; gradient_backend=:analytic,
    max_iter=nothing, gtol=1e-8, stats=OptimizationStats(), max_seconds=Inf)
rows = NamedTuple[]
power_rows = NamedTuple[]
for m in 2:max_bits
    solver = UnitaryDecoder(m, 1; cost_type=NORM, random_seed=42)
    initial = initialize_point(solver)
    D._optimize(solver, initial.U, initial.params; gradient_backend=:analytic,
        max_iter=2, gtol=1e-8, stats=OptimizationStats(), max_seconds=Inf)
    stats = OptimizationStats()
    measured = @timed D._optimize(solver, initial.U, initial.params;
        gradient_backend=:analytic, max_iter=nothing, gtol=1e-8,
        stats=stats, max_seconds=Inf)
    if measured.compile_time>0 || measured.recompile_time>0
        stats = OptimizationStats()
        measured = @timed D._optimize(solver, initial.U, initial.params;
            gradient_backend=:analytic, max_iter=nothing, gtol=1e-8,
            stats=stats, max_seconds=Inf)
    end
    measured.compile_time==0 && measured.recompile_time==0 ||
        error("Compilation remains in a formal measurement.")
    result = measured.value
    # Save the measured design after the timed solve for the paper's power maps.
    power = normalized_power(D._decoder_output(solver, result.U, result.params))
    append!(power_rows, [(; bits=m, output_port=i, input_word=j, normalized_power=power[i,j])
        for j in 1:solver.n for i in 1:solver.n])
    write_csv(joinpath(OUT, "scaling_power.csv"), power_rows)
    analysis = analyze_result(solver, result)
    gradient = gradient_norm(solver, result; gradient_backend=:analytic)
    push!(rows, (; bits=m, ports=solver.n, num_bias_params=solver.num_params, backend="analytic", seed=42,
        elapsed_seconds=measured.time, allocated_bytes=measured.bytes,
        iterations=stats.iterations, cost_evaluations=stats.cost_evaluations,
        gradient_evaluations=stats.gradient_evaluations, retractions=stats.retractions,
        compile_seconds=measured.compile_time, recompile_seconds=measured.recompile_time,
        max_iter="none", max_seconds=Inf, requested_gtol=1e-8,
        stop_reason=string(stats.stop_reason), cost=analysis.cost,
        final_gradient_norm=gradient, gradient_tolerance_met=gradient<1e-8,
        unitarity_residual=norm(result.U'*result.U-I)))
    write_csv(joinpath(OUT, "extended_scalability.csv"), rows)
    println(last(rows))
    flush(stdout)
end
