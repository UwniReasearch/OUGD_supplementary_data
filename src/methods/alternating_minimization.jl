# Alternating minimisation. Each block is solved to its recorded gradient tolerance.
# Inner and outer iterations have no iteration or time cap.

function _alternating_parameter_step(cost_fn::Function,
    trial_evaluator::Union{Nothing,Function}, precision_evaluator::Union{Nothing,Function},
    U::Matrix, params::Vector, gradient_params::Vector, current_cost::Float64;
    α_init::Float64=1.0, c::Float64=1e-4, τ::Float64=0.5, max_iter::Int=50)
    evaluate_trial = if isnothing(trial_evaluator)
        (trial_params, required_difference) -> begin
            value = cost_fn(U, trial_params)
            return value, isfinite(value) && value <= current_cost + required_difference
        end
    else
        evaluate = trial_evaluator(U, params, current_cost)
        (trial_params, required_difference) -> evaluate(U, trial_params, required_difference)
    end
    slope = -real(dot(gradient_params, gradient_params))
    α = α_init
    for _ in 1:max_iter
        trial_params = params - α * gradient_params
        trial_cost, accepted = evaluate_trial(trial_params, c * α * slope)
        if accepted
            return (; accepted=true, α, params=trial_params, cost=trial_cost)
        end
        α *= τ
    end
    if !isnothing(precision_evaluator)
        evaluate_precisely = precision_evaluator(U, params, current_cost)
        precise_alpha = α_init
        for _ in 1:max_iter
            trial_params = params - precise_alpha * gradient_params
            trial_cost, accepted = evaluate_precisely(U, trial_params, c * precise_alpha * slope)
            if accepted
                return (; accepted=true, α=precise_alpha, params=trial_params, cost=trial_cost)
            end
            precise_alpha *= τ
        end
    end
    # Preserve the existing solver's final-step rule after the same 50 checks.
    final_params = params - α * gradient_params
    return (; accepted=false, α, params=final_params, cost=cost_fn(U, final_params))
end

"""
    _alternating_bias_value_gradient(solver, U, params, stats)

Bias-only evaluation of the same analytic objective and derivative. Reuses the
output-gradient function and the original UrefᴴG1 formula, without constructing
an unused ambient U gradient. One invocation is one cost and gradient evaluation.
"""
function _alternating_bias_value_gradient(solver::UnitarySolver, U::Matrix,
    params::Vector, stats::OptimizationStats)
    stats.cost_evaluations += 1
    stats.gradient_evaluations += 1
    cfg = solver.decoder_config
    value, output_gradient = _decoder_cost_output_gradient(
        _decoder_output(solver, U, params), cfg.cost_type)
    if isempty(params)
        return value, similar(params, 0)
    end
    reference = view(U, :, cfg.num_bits+1:cfg.num_bits+length(params))
    gradient = reference' * vec(sum(output_gradient; dims=2))
    eltype(params) <: Real && (gradient = real.(gradient))
    return value, gradient
end

function alternating_minimize(solver::UnitarySolver, U0::Matrix, params0::Vector;
    gtol::Float64=1e-8, subproblem_gtol::Float64=gtol/sqrt(2),
    stats::OptimizationStats=OptimizationStats(), record::Bool=false,
    progress_callback::Union{Nothing,Function}=nothing, progress_every::Int=10)
    isfinite(gtol) && gtol > 0 || throw(ArgumentError("gtol must be finite and positive."))
    isfinite(subproblem_gtol) && 0 < subproblem_gtol <= gtol/sqrt(2) ||
        throw(ArgumentError("subproblem_gtol must be positive and at most gtol/sqrt(2)."))
    progress_every > 0 || throw(ArgumentError("progress_every must be positive."))
    _reset_stats!(stats)
    cost_fn = _cost_evaluator(solver, stats)
    trial_evaluator = _armijo_trial_evaluator(solver, stats)
    precision_evaluator = _armijo_precision_evaluator(solver, stats)
    backend, value_gradient = _gradient_evaluator(solver, :analytic, cost_fn, stats)
    stats.gradient_backend = backend
    retract = (U, direction) -> begin
        stats.retractions += 1
        _retract_exp(U, direction)
    end
    U, params = copy(U0), copy(params0)
    fixed_bias_gradient = zero(params)
    history = record ? NamedTuple[] : nothing
    started = time_ns()
    cycles = unitary_updates = bias_updates = line_search_fallbacks = 0
    joint_gradient_evaluations = unitary_gradient_evaluations = bias_gradient_evaluations = 0
    report = (phase, value, gradient_size, done) -> begin
        isnothing(progress_callback) || progress_callback((; cycles, phase,
            block_updates=unitary_updates+bias_updates, unitary_updates, bias_updates,
            cost=value, gradient_norm=gradient_size,
            joint_gradient_norm=phase == :joint ? gradient_size : NaN,
            elapsed_seconds=(time_ns()-started)/1e9, done))
    end
    while true
        current_cost, ambient_gradient, parameter_gradient = value_gradient(U, params)
        stats.gradient_evaluations += 1
        joint_gradient_evaluations += 1
        tangent_gradient = _rgrad_unitary(U, ambient_gradient)
        unitary_norm = norm(tangent_gradient)
        bias_norm = norm(parameter_gradient)
        stats.last_gradient_norm = sqrt(real(dot(tangent_gradient, tangent_gradient)) +
                                        real(dot(parameter_gradient, parameter_gradient)))
        if record && cycles == 0
            push!(history, (; cycle=0, block=:initial, U=copy(U), params=copy(params), cost=current_cost))
        end
        done = stats.last_gradient_norm < gtol
        report(:joint, current_cost, stats.last_gradient_norm, done)
        if done
            stats.iterations = cycles
            stats.stop_reason = :gradient_tol
            stats.elapsed_seconds = (time_ns() - started) / 1e9
            result = (; U, params, stop_reason=:gradient_tol, cycles,
                block_updates=unitary_updates+bias_updates, unitary_updates,
                bias_updates, line_search_fallbacks, subproblem_gtol,
                joint_gradient_evaluations, unitary_gradient_evaluations,
                bias_gradient_evaluations)
            return result, stats, history
        end
        next_cycle = cycles + 1
        # Solve the U subproblem while holding params exactly fixed.
        while unitary_norm >= subproblem_gtol
            step = _armijo_cached(cost_fn, U, params, tangent_gradient,
                fixed_bias_gradient, retract, current_cost; trial_evaluator, precision_evaluator)
            U = step.U
            unitary_updates += 1
            line_search_fallbacks += !step.accepted
            record && push!(history, (; cycle=next_cycle, block=:unitary,
                U=copy(U), params=copy(params), cost=step.cost))
            current_cost, ambient_gradient, parameter_gradient = value_gradient(U, params)
            stats.gradient_evaluations += 1
            unitary_gradient_evaluations += 1
            tangent_gradient = _rgrad_unitary(U, ambient_gradient)
            unitary_norm = norm(tangent_gradient)
            if unitary_updates % progress_every == 0
                report(:unitary, current_cost, unitary_norm, false)
            end
        end
        # The last full evaluation already supplies the bias derivative at this U.
        bias_norm = norm(parameter_gradient)
        while bias_norm >= subproblem_gtol
            step = _alternating_parameter_step(cost_fn, trial_evaluator,
                precision_evaluator, U, params, parameter_gradient, current_cost)
            params = step.params
            bias_updates += 1
            line_search_fallbacks += !step.accepted
            record && push!(history, (; cycle=next_cycle, block=:bias,
                U=copy(U), params=copy(params), cost=step.cost))
            current_cost, parameter_gradient = _alternating_bias_value_gradient(solver, U, params, stats)
            bias_gradient_evaluations += 1
            bias_norm = norm(parameter_gradient)
            if bias_updates % progress_every == 0
                report(:bias, current_cost, bias_norm, false)
            end
        end
        cycles = next_cycle
    end
end
