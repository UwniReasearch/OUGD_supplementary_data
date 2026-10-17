include("common.jl")
const METHOD_GTOL = 1e-8
const METHOD_NAMES = ("rgd_joint", "euclidean_gradient_descent", "alternating_minimization")

field_or(result, name, default=0) = hasproperty(result,name) ? getproperty(result,name) : default

function method_solve(solver, method, seed)
    solver.num_params == 1 || error("The paper uses exactly one complex bias parameter.")
    initial = initialize_point(solver; random_seed=seed)
    stats = OptimizationStats()
    # Progress reporting is infrequent; it never changes the stopping criterion.
    last_report = Ref(time_ns())
    callback = row -> begin
        now = time_ns()
        if (now-last_report[])/1e9 >= 30
            count = hasproperty(row,:iteration) ? row.iteration : row.cycles
            @printf("%s m=%d seed=%d updates/cycles=%d J=%.12g gradient=%.5g\n",
                method,solver.decoder_config.num_bits,seed,count,row.cost,row.gradient_norm)
            flush(stdout)
            last_report[] = now
        end
    end
    if method=="rgd_joint"
        result = _optimize(solver,initial.U,initial.params;
            gradient_backend=:analytic,retraction=:exp,max_iter=nothing,
            gtol=METHOD_GTOL,stats=stats,max_seconds=Inf)
    elseif method=="euclidean_gradient_descent"
        result,stats,_ = euclidean_gradient_descent(solver,initial.U,initial.params;
            gtol=METHOD_GTOL,stats=stats,progress_callback=callback)
    elseif method=="alternating_minimization"
        result,stats,_ = alternating_minimize(solver,initial.U,initial.params;
            gtol=METHOD_GTOL,subproblem_gtol=METHOD_GTOL/sqrt(2),
            stats=stats,progress_callback=callback,progress_every=1)
    else
        error("Unknown comparison method: $method")
    end
    return (;result,stats)
end

function method_verify(solver,result)
    Y = _decoder_output(solver,result.U,result.params)
    cost = solver.cost_fn(Y)
    gradient = gradient_norm(solver,result;gradient_backend=:analytic)
    return (;cost,
        final_gradient_norm=gradient,gradient_tolerance_met=gradient<METHOD_GTOL,
        unitarity_residual=norm(result.U'*result.U-I),
        bias_norm=norm(result.params))
end

function method_measure(solver,method,seed,repetition)
    measured = @timed method_solve(solver,method,seed)
    if measured.compile_time>0 || measured.recompile_time>0
        println("First use still compiled; excluding this complete call and repeating the same initial point.")
        flush(stdout)
        measured = @timed method_solve(solver,method,seed)
    end
    measured.compile_time==0 && measured.recompile_time==0 ||
        error("Compilation remains in a formal measurement.")
    result,stats = measured.value.result,measured.value.stats
    verification = @timed method_verify(solver,result)
    if verification.compile_time>0 || verification.recompile_time>0
        verification = @timed method_verify(solver,result)
    end
    verification.compile_time==0 && verification.recompile_time==0 ||
        error("Compilation remains in verification.")
    metrics = verification.value
    metrics.gradient_tolerance_met == (result.stop_reason==:gradient_tol) ||
        error("Stop reason and original joint-gradient verification disagree.")
    unit = method=="alternating_minimization" ? "outer_cycles" :
        method=="euclidean_gradient_descent" ? "coordinate_updates" : "joint_updates"
    row = (bits=solver.decoder_config.num_bits,ports=solver.n,num_bias_params=solver.num_params,method=method,
        seed=seed,repetition=repetition,solve_seconds=measured.time,
        verification_seconds=verification.time,allocated_bytes=measured.bytes,
        gc_seconds=measured.gctime,compile_seconds=measured.compile_time,
        recompile_seconds=measured.recompile_time,iterations=stats.iterations,
        iteration_unit=unit,cost_evaluations=stats.cost_evaluations,
        gradient_evaluations=stats.gradient_evaluations,
        matrix_exponentials=stats.retractions,
        eigendecompositions=field_or(result,:eigendecomposition_evaluations),
        schur_decompositions=field_or(result,:schur_decompositions),
        unitary_updates=field_or(result,:unitary_updates),
        bias_updates=field_or(result,:bias_updates),
        joint_gradient_evaluations=field_or(result,:joint_gradient_evaluations,stats.gradient_evaluations),
        unitary_gradient_evaluations=field_or(result,:unitary_gradient_evaluations),
        bias_gradient_evaluations=field_or(result,:bias_gradient_evaluations),
        initial_reconstruction_error=field_or(result,:initial_reconstruction_error,0.0),
        double_step_checks=field_or(result,:double_step_checks,NaN),
        precision_step_checks=field_or(result,:precision_step_checks,NaN),
        extended_step_checks=field_or(result,:extended_step_checks,NaN),
        max_iter="none",max_seconds=Inf,requested_gtol=METHOD_GTOL,
        subproblem_gtol=method=="alternating_minimization" ? METHOD_GTOL/sqrt(2) : NaN,
        stop_reason=string(result.stop_reason))
    return merge(row,metrics,(unaccepted_armijo_updates=field_or(result,:unaccepted_armijo_updates,NaN),))
end

function method_summary(rows,path)
    summary = NamedTuple[]
    for bits in sort(unique(r.bits for r in rows)),method in METHOD_NAMES
        selected = filter(r->r.bits==bits && r.method==method,rows)
        isempty(selected) && continue
        completed = filter(r->r.gradient_tolerance_met,selected)
        push!(summary,(bits=bits,ports=2^bits,method=method,runs=length(selected),
            successes=length(completed),seeds=length(unique(r.seed for r in selected)),
            solve_median_seconds=isempty(completed) ? NaN : median(r.solve_seconds for r in completed),
            solve_min_seconds=isempty(completed) ? NaN : minimum(r.solve_seconds for r in completed),
            solve_max_seconds=isempty(completed) ? NaN : maximum(r.solve_seconds for r in completed),
            median_cost_evaluations=median(r.cost_evaluations for r in selected),
            median_gradient_evaluations=median(r.gradient_evaluations for r in selected),
            maximum_final_gradient=maximum(r.final_gradient_norm for r in selected),
            maximum_unitarity_residual=maximum(r.unitarity_residual for r in selected)))
    end
    write_csv(path,summary)
    return summary
end


bits = parse.(Int, split(option("--bits", "2,3,4,5,6,7,8,9"), ','))
seeds = parse.(Int, split(option("--seeds", "42"), ','))
repeats = parse(Int, option("--repeats", "3"))
prefix = option("--prefix", "method_comparison")
all(b -> 2 <= b <= 9, bits) && repeats > 0 || error("Invalid experiment size.")
occursin(r"^[a-z0-9_]+$", prefix) || error("Invalid output prefix.")
for method in METHOD_NAMES
    warm = UnitaryDecoder(2, 1; cost_type=NORM, random_seed=42)
    method_verify(warm, method_solve(warm, method, 42).result)
end
rows = NamedTuple[]
for m in bits, seed in seeds, repetition in 1:repeats
    solver = UnitaryDecoder(m, 1; cost_type=NORM, random_seed=seed)
    offset = (repetition-1) % length(METHOD_NAMES)
    methods = [METHOD_NAMES[offset+1:end]...; METHOD_NAMES[1:offset]...]
    for method in methods
        push!(rows, method_measure(solver, method, seed, repetition))
        write_csv(joinpath(OUT, prefix*".csv"), rows)
        println(last(rows))
        flush(stdout)
    end
end
method_summary(rows, joinpath(OUT, prefix*"_summary.csv"))
