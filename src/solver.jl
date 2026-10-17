# Gradient selection, work counters, initialization and optimisation.

Base.@kwdef mutable struct OptimizationStats
    iterations::Int = 0
    cost_evaluations::Int = 0
    gradient_evaluations::Int = 0
    retractions::Int = 0
    last_gradient_norm::Float64 = NaN
    elapsed_seconds::Float64 = 0.0
    stop_reason::Symbol = :not_started
    gradient_backend::Symbol = :unresolved
end

function _reset_stats!(stats::OptimizationStats)
    stats.iterations = stats.cost_evaluations = stats.gradient_evaluations = stats.retractions = 0
    stats.last_gradient_norm = NaN
    stats.elapsed_seconds = 0.0
    stats.stop_reason = :not_started
    stats.gradient_backend = :unresolved
    return stats
end

"""
    _resolve_gradient_backend(solver, requested) -> Symbol

Select the derivative implementation before optimisation. `:auto` uses analytic
derivatives for NORM and EXTINCTION decoders and central finite differences for
other objectives. An explicit request takes precedence. The analytic interface
rejects unsupported objectives with ArgumentError.
"""
function _resolve_gradient_backend(solver::UnitarySolver, requested::Symbol)
    requested in (:auto, :finite_difference, :analytic) ||
        throw(ArgumentError("gradient_backend must be :auto, :finite_difference or :analytic."))
    requested != :auto && return requested
    cfg = solver.decoder_config
    return !isnothing(cfg) && cfg.cost_type in (NORM, EXTINCTION) ?
        :analytic : :finite_difference
end

# Evaluate the output using the input structure, independently of differentiation.
function _cost_evaluator(solver::UnitarySolver, stats::Union{Nothing,OptimizationStats}=nothing)
    output = isnothing(solver.decoder_config) ?
        ((U, params) -> U * solver.build_input(params)) :
        ((U, params) -> _decoder_output(solver, U, params))
    return (U, params) -> begin
        isnothing(stats) || (stats.cost_evaluations += 1)
        solver.cost_fn(output(U, params))
    end
end

# Both derivative implementations return (cost, ambient_U, gradient_params).
# Select the implementation here so the optimisation loop needs no backend branch.
function _gradient_evaluator(solver::UnitarySolver, requested::Symbol, cost_fn::Function,
    stats::Union{Nothing,OptimizationStats}=nothing)
    backend = _resolve_gradient_backend(solver, requested)
    if backend == :analytic
        isnothing(solver.decoder_config) &&
            throw(ArgumentError("Analytic differentiation requires UnitaryDecoder."))
        solver.decoder_config.cost_type in (NORM, EXTINCTION) ||
            throw(ArgumentError("Analytic differentiation supports NORM and EXTINCTION."))
        evaluate = (U, params) -> begin
            isnothing(stats) || (stats.cost_evaluations += 1)
            _decoder_ambient_value_gradient(solver, U, params)
        end
    else
        evaluate = (U, params) ->
            _finite_difference_value_gradient(solver, U, params, cost_fn)
    end
    return backend, evaluate
end

# Select objective comparisons independently of the derivative implementation.
# For NORM, cache the current power table and reuse each new output and power table.
function _armijo_trial_evaluator(solver::UnitarySolver,
    stats::Union{Nothing,OptimizationStats}=nothing)
    cfg = solver.decoder_config
    isnothing(cfg) && return nothing
    # H uses the same decrease condition evaluated at higher precision.
    # Direct Float64 comparisons can reject a decrease smaller than one ulp.
    cfg.cost_type == EXTINCTION && return _armijo_precision_evaluator(solver, stats)
    cfg.cost_type != NORM && return nothing
    return (U, params, _current_cost) -> begin
        current_power = _normalized_power_columns(abs2.(_decoder_output(solver, U, params)))
        (trial_U, trial_params, required_difference) -> begin
            trial_power = _normalized_power_columns(abs2.(_decoder_output(solver, trial_U, trial_params)))
            isnothing(stats) || (stats.cost_evaluations += 1)
            trial_cost = _norm_cost_from_normalized_power(trial_power, solver.n)
            difference = compute_norm_cost_difference(current_power, trial_power)
            accepted = isfinite(trial_cost) && isfinite(difference) && difference <= required_difference
            return trial_cost, accepted
        end
    end
end

# Reevaluate the same objective at higher precision; updates remain in Float64.
# J uses this after double-precision step selection fails; H uses it for every check.
# Recompute from U and params rather than converting an already rounded power table.
function _armijo_precision_evaluator(solver::UnitarySolver,
    stats::Union{Nothing,OptimizationStats}=nothing)
    cfg = solver.decoder_config
    (isnothing(cfg) || !(cfg.cost_type in (NORM, EXTINCTION))) && return nothing
    objective = cfg.cost_type == NORM ?
        (power -> _norm_cost_from_normalized_power(_normalized_power_columns(power), solver.n)) :
        (power -> _cr_cost_from_power(power, solver.n))
    precise_cost = (U, params) -> setprecision(BigFloat, 256) do
        isnothing(stats) || (stats.cost_evaluations += 1)
        output = _decoder_output(solver, Complex{BigFloat}.(U), Complex{BigFloat}.(params))
        objective(abs2.(output))
    end
    return (U, params, _current_cost) -> begin
        reference = precise_cost(U, params)
        (new_U, new_params, required_difference) -> setprecision(BigFloat, 256) do
            value = precise_cost(new_U, new_params)
            accepted = isfinite(value) && value-reference <= BigFloat(required_difference)
            return Float64(value), accepted
        end
    end
end

function initialize_point(solver::UnitarySolver{T}; random_seed::Union{Nothing,Int}=nothing) where T
    Random.seed!(something(random_seed, solver.random_seed))
    U0 = Matrix(qr(randn(ComplexF64, solver.n, solver.n)).Q)
    params0 = if solver.num_params == 0
        T[]
    elseif T <: Real
        ComplexF64.(randn(Float64, solver.num_params))
    else
        randn(ComplexF64, solver.num_params)
    end
    return (U=U0, params=params0)
end

function _optimize(solver::UnitarySolver, U0::Matrix, params0::Vector;
    max_iter::Union{Nothing,Int}=200, gtol::Float64=1e-8, record::Bool=false,
    retraction::Symbol=:exp, gradient_backend::Symbol=:auto,
    stats::Union{Nothing,OptimizationStats}=nothing, max_seconds::Float64=Inf)
    retraction in (:exp, :qr) || throw(ArgumentError("retraction must be :exp or :qr."))
    isnothing(stats) || _reset_stats!(stats)
    cost_fn = _cost_evaluator(solver, stats)
    trial_evaluator = _armijo_trial_evaluator(solver, stats)
    precision_evaluator = _armijo_precision_evaluator(solver, stats)
    backend, value_gradient = _gradient_evaluator(solver, gradient_backend, cost_fn, stats)
    isnothing(stats) || (stats.gradient_backend = backend)
    raw_retract = retraction == :qr ? _retract_qr : _retract_exp
    retract = (U, ξ) -> begin
        isnothing(stats) || (stats.retractions += 1)
        raw_retract(U, ξ)
    end
    U, params = copy(U0), copy(params0)
    history = record ? Tuple{Any,Float64}[] : nothing
    started = time_ns()
    accepted_steps = 0
    stop_reason = :not_started
    while true
        current_cost, ambient_gradient, parameter_gradient = value_gradient(U, params)
        tangent_gradient = _rgrad_unitary(U, ambient_gradient)
        gradient_size = sqrt(real(dot(tangent_gradient, tangent_gradient)) +
                             real(dot(parameter_gradient, parameter_gradient)))
        if !isnothing(stats)
            stats.gradient_evaluations += 1
            stats.last_gradient_norm = gradient_size
        end
        if gradient_size < gtol
            stop_reason = :gradient_tol
            break
        elseif !isnothing(max_iter) && accepted_steps >= max_iter
            stop_reason = :max_iter
            break
        elseif (time_ns() - started) / 1e9 >= max_seconds
            stop_reason = :time_budget
            break
        end
        step = _armijo_cached(cost_fn, U, params, tangent_gradient,
            parameter_gradient, retract, current_cost; trial_evaluator, precision_evaluator)
        U, params = step.U, step.params
        accepted_steps += 1
        record && push!(history, ((; U=copy(U), params=copy(params)), step.cost))
    end
    if !isnothing(stats)
        stats.iterations = accepted_steps
        stats.stop_reason = stop_reason
        stats.elapsed_seconds = (time_ns() - started) / 1e9
    end
    result = (; U, params, stop_reason)
    return record ? (result, history) : result
end

function optimize!(solver::UnitarySolver; verbose::Bool=true,
    retraction::Symbol=:exp, max_iter::Union{Nothing,Int}=200, gtol::Float64=1e-8,
    gradient_backend::Symbol=:auto,
    stats::Union{Nothing,OptimizationStats}=nothing, max_seconds::Float64=Inf)
    Random.seed!(solver.random_seed)
    p0      = initialize_point(solver)
    cost_fn = (V, p) -> solver.cost_fn(V * solver.build_input(p))

    verbose && println("Initial objective value: $(round(cost_fn(p0.U, p0.params), digits=8))")

    result = _optimize(solver, p0.U, p0.params; retraction, max_iter, gtol,
        gradient_backend, stats, max_seconds)
    solver.last_result = result
    return result
end

function get_output(solver::UnitarySolver, result)
    U = result.U
    X = solver.build_input(result.params)
    return U * X, X
end

# ============================================================================
# Result analysis and display
# ============================================================================

"""
    analyze_result(solver, result) -> NamedTuple

Return `(cost, U, X, Y, params)` for the computed result.

Paper metrics are obtained with paper_metrics(normalized_power(analysis.Y)).
"""
function analyze_result(solver::UnitarySolver, result)
    U    = result.U
    X    = solver.build_input(result.params)
    Y    = U * X
    cost = solver.cost_fn(Y)
    p    = isempty(result.params) ? nothing : result.params
    return (; cost, U, X, Y, params=p)
end

"""
    gradient_norm(solver, result; gradient_backend=:auto) -> Float64

Compute the product-manifold gradient norm using the same derivative selection
and projection as the optimiser. A small norm indicates approximate stationarity;
it does not distinguish a local minimum, a saddle point or a global minimum.
"""
function gradient_norm(solver::UnitarySolver, result; gradient_backend::Symbol=:auto)
    cost_fn = _cost_evaluator(solver)
    _, value_gradient = _gradient_evaluator(solver, gradient_backend, cost_fn)
    _, ambient_gradient, parameter_gradient = value_gradient(result.U, result.params)
    tangent_gradient = _rgrad_unitary(result.U, ambient_gradient)
    return sqrt(real(dot(tangent_gradient, tangent_gradient)) +
                real(dot(parameter_gradient, parameter_gradient)))
end

