# Ordinary Euclidean gradient descent in fixed real Hermitian coordinates.
# Loaded by Journal2Decoder after its shared solver and coordinate definitions.


"""
One coordinate-gradient step with the same Armijo constants and step-length
sequence as the existing Riemannian solver: alpha=1, c=1e-4, halving, 50 double
precision candidates followed (for NORM) by the same 50 high-precision candidates.
If neither pass selects a step, use its final alpha=2^-50 exactly as the core.
That last update is explicitly recorded as not accepted by Armijo. If its
coordinate vector cannot change in Float64, return moved=false rather than
repeat the identical point indefinitely. No extra step-length tuning is used.
"""
function _euclidean_gradient_armijo_step(cost_fn::Function,
    trial_factory::Union{Nothing,Function}, precision_factory::Union{Nothing,Function},
    state, x::Vector{Float64}, g::Vector{Float64}, evaluate_state::Function,
    current_cost::Float64)
    slope = -dot(g,g)
    double_evaluator = if isnothing(trial_factory)
        (U, params, difference) -> begin
            value = cost_fn(U, params)
            return value, isfinite(value) && value <= current_cost+difference
        end
    else
        trial_factory(state.U, state.params, current_cost)
    end
    attempt = (alpha, evaluator) -> begin
        point = x-alpha*g
        point == x && return nothing
        candidate = evaluate_state(point)
        cost, accepted = evaluator(candidate.U, candidate.params, 1e-4*alpha*slope)
        return (; point, state=candidate, cost, accepted, step_length=alpha)
    end
    double_step_checks = 0
    precision_step_checks = 0
    alpha = 1.0
    for _ in 1:50
        double_step_checks += 1
        step = attempt(alpha, double_evaluator)
        if !isnothing(step) && step.accepted
            return merge(step, (; moved=true, double_step_checks, precision_step_checks))
        end
        alpha *= 0.5
    end
    if !isnothing(precision_factory)
        precise_evaluator = precision_factory(state.U, state.params, current_cost)
        alpha = 1.0
        for _ in 1:50
            precision_step_checks += 1
            step = attempt(alpha, precise_evaluator)
            if !isnothing(step) && step.accepted
                return merge(step, (; moved=true, double_step_checks, precision_step_checks))
            end
            alpha *= 0.5
        end
    end
    # This is the same final step returned by the existing core when no tested
    # candidate passed. It is an update, not an Armijo-accepted step.
    point = x-alpha*g
    if point == x
        return (; point=x, state, cost=current_cost, accepted=false, moved=false,
            step_length=alpha, double_step_checks, precision_step_checks)
    end
    candidate = evaluate_state(point)
    return (; point, state=candidate, cost=cost_fn(candidate.U, candidate.params),
        accepted=false, moved=true, step_length=alpha,
        double_step_checks, precision_step_checks)
end

"""
    euclidean_gradient_descent(solver, U0, params0; gtol=1e-8,
        stats=OptimizationStats(), record=false, progress_callback=nothing)
        -> (result, stats, history)

Ordinary Euclidean gradient descent on the fixed n² real coordinates of a
Hermitian H and on all original bias coordinates, using U=exp(iH). Every update
is x_new=x-alpha*gradient_x. Both methods thus use first-order steepest descent
and the same Armijo policy, with different coordinates for the unitary variable.
This routine uses neither momentum nor a Hessian approximation.

The input U0 is converted by its general unitary Schur decomposition, and
params0 is unchanged. Only the shared original (U,a) joint gradient norm < gtol
counts as convergence. There is no time or iteration limit. Coordinate-gradient
zero alone is chart_stationarity; unchanged Float64 coordinates after the
core's final fallback step are numerical_stagnation if physical gtol is unmet.

stats.iterations counts all actual coordinate updates, including any core-policy
fallback updates; result.unaccepted_armijo_updates counts those separately.
stats.retractions=0. result.eigendecomposition_evaluations counts the Hermitian
spectral evaluations; schur_decompositions=1. extended_step_checks=0 because
the step lengths exactly follow the existing core, without further shrinking.
Timing includes the initial Schur conversion and final gradient check.
history and progress_callback record iteration,cost,gradient_norm,
coordinate_gradient_norm,step_length,armijo_accepted, including iteration 0.
The input solver, U0 and params0 are not mutated.
"""
function euclidean_gradient_descent(solver::UnitarySolver, U0::Matrix,
    params0::Vector; gtol::Float64=1e-8, stats::OptimizationStats=OptimizationStats(),
    record::Bool=false, progress_callback::Union{Nothing,Function}=nothing)
    cfg = solver.decoder_config
    !isnothing(cfg) && cfg.cost_type in (NORM, EXTINCTION) ||
        throw(ArgumentError("Euclidean-coordinate gradient descent needs analytic NORM/EXTINCTION."))
    gtol > 0 || throw(ArgumentError("Gradient tolerance must be positive."))
    size(U0) == (solver.n,solver.n) || throw(DimensionMismatch("U0 has wrong size."))
    length(params0) == solver.num_params || throw(DimensionMismatch("params0 has wrong size."))
    _reset_stats!(stats)
    stats.gradient_backend = :analytic
    started = time_ns()
    x = _unitary_initial_coordinates(U0,params0)
    n, q = solver.n, length(params0)
    complex_params = !(eltype(params0) <: Real)
    decompositions = 0
    evaluate_state = point -> begin
        decompositions += 1
        _unitary_coordinate_state(point,n,q,complex_params)
    end
    state = evaluate_state(x)
    current_cost,g,physical_norm = _unitary_coordinate_value_gradient(solver,state,stats)
    initial_reconstruction_error = norm(state.U-U0)/norm(U0)
    cost_fn = _cost_evaluator(solver,stats)
    trial_factory = _armijo_trial_evaluator(solver,stats)
    precision_factory = _armijo_precision_evaluator(solver,stats)
    history = NamedTuple[]
    iteration = 0
    double_step_checks = 0
    precision_step_checks = 0
    unaccepted_armijo_updates = 0
    last_step_length = 0.0
    last_armijo_accepted = true
    reason = :not_started
    while true
        row = (; iteration, cost=current_cost, gradient_norm=physical_norm,
            coordinate_gradient_norm=norm(g), step_length=last_step_length,
            armijo_accepted=last_armijo_accepted)
        record && push!(history,row)
        isnothing(progress_callback) || progress_callback(row)
        stats.last_gradient_norm = physical_norm
        if physical_norm < gtol
            reason = :gradient_tol
            break
        elseif !isfinite(current_cost) || !all(isfinite,g) || !isfinite(physical_norm)
            reason = :nonfinite
            break
        elseif iszero(norm(g))
            reason = :chart_stationarity
            break
        end
        step = _euclidean_gradient_armijo_step(cost_fn,trial_factory,precision_factory,
            state,x,g,evaluate_state,current_cost)
        double_step_checks += step.double_step_checks
        precision_step_checks += step.precision_step_checks
        if !step.moved
            reason = :numerical_stagnation
            break
        end
        x,state = step.point,step.state
        current_cost,g,physical_norm = _unitary_coordinate_value_gradient(solver,state,stats)
        iteration += 1
        last_step_length = step.step_length
        last_armijo_accepted = step.accepted
        unaccepted_armijo_updates += !step.accepted
    end
    stats.iterations = iteration
    stats.stop_reason = reason
    stats.elapsed_seconds = (time_ns()-started)/1e9
    result = (; U=state.U, params=state.params, stop_reason=reason,
        coordinate_gradient_norm=norm(g), eigendecomposition_evaluations=decompositions,
        schur_decompositions=1, initial_reconstruction_error,
        double_step_checks, precision_step_checks, extended_step_checks=0,
        unaccepted_armijo_updates)
    return result,stats,history
end
