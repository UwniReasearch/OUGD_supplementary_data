include("common.jl")

# Compare the two stated objectives from identical single-bias initial points.
backend = :analytic
bits = parse.(Int, split(option("--bits", "2,3,4,5,6"), ','))
seed = parse(Int, option("--seed", "42"))
all(m -> 2 <= m <= 9, bits) || error("Expected bit widths between 2 and 9.")
rows, pairs = NamedTuple[], NamedTuple[]
for m in bits
    solutions = Dict{String,Any}()
    for (name, cost_type) in (("NORM", NORM), ("EXTINCTION", EXTINCTION))
        solver = UnitaryDecoder(m, 1; cost_type, random_seed=seed)
        println("START bits=",m," bias_parameters=1 objective=",name," seed=",seed)
        flush(stdout)
        initial = initialize_point(solver)
        stats = OptimizationStats()
        result = D._optimize(solver, initial.U, initial.params;
            gradient_backend=backend, retraction=:exp, max_iter=nothing,
            max_seconds=Inf, gtol=1e-8, stats)
        metrics = verify(solver, result; gradient_backend=backend)
        power = normalized_power(D._decoder_output(solver, result.U, result.params))
        solutions[name] = (; power, initial, metrics)
        push!(rows, merge((; bits=m, ports=2^m, num_bias_params=solver.num_params, seed, objective=name,
            gradient_backend=string(backend), max_iter="none", max_seconds=Inf,
            requested_gtol=1e-8, iterations=stats.iterations,
            stop_reason=string(result.stop_reason)), metrics))
        write_csv(joinpath(OUT,"objective_comparison_runs.csv"),rows)
        println(last(rows))
        flush(stdout)
    end
    a, b = solutions["NORM"], solutions["EXTINCTION"]
    push!(pairs, (; bits=m, num_bias_params=1, seed, max_iter="none", max_seconds=Inf,
        requested_gtol=1e-8,
        initial_U_max_difference=maximum(abs.(a.initial.U-b.initial.U)),
        initial_bias_max_difference=maximum(abs.(a.initial.params-b.initial.params)),
        P_pair_Frobenius_difference=norm(a.power-b.power),
        P_pair_max_difference=maximum(abs.(a.power-b.power)),
        J_pair_difference=a.metrics.J-b.metrics.J,
        H_pair_difference=a.metrics.H-b.metrics.H))
    write_csv(joinpath(OUT, "objective_comparison_pairs.csv"), pairs)
end
write_csv(joinpath(OUT, "objective_comparison_runs.csv"), rows)
write_csv(joinpath(OUT, "objective_comparison_pairs.csv"), pairs)
foreach(println, rows)
