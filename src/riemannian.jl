# Tangent projection, exponential update and Armijo step selection.
# The numerical rules are unchanged from the measured implementation.

_rgrad_unitary(U::Matrix, E::Matrix) = U * ((U' * E - E' * U) / 2)
function _rgrad_unitary(U::Matrix, E::NamedTuple{(:left, :right)})
    L, C = E.left, E.right
    left_times_U = L' * U
    projected_left = U * left_times_U'
    return (projected_left * C - (U * C') * left_times_U) / 2
end

# For ξ=U*A, A is skew-Hermitian. Enforce this symmetry to remove rounding error.
function _retract_exp(U::Matrix, ξ::Matrix)
    generator = U' * ξ
    return U * exp((generator - generator') / 2)
end
_retract_exp_stable(U::Matrix, ξ::Matrix) = _retract_exp(U, ξ)
_retract_qr(U::Matrix, ξ::Matrix) = Matrix(qr(U + ξ).Q)
_retract_unitary(U::Matrix, ξ::Matrix, ::Val{:exp}) = _retract_exp(U, ξ)
_retract_unitary(U::Matrix, ξ::Matrix, ::Val{:qr}) = _retract_qr(U, ξ)
_retract_unitary(U::Matrix, ξ::Matrix) = _retract_exp(U, ξ)

"""
    _armijo_cached(cost_fn, U, params, rgrad_U, grad_p, retract, f0; ...)

Select a step by Armijo backtracking on the unitary and input-parameter product
space. Complex gradients include both real and imaginary derivatives, so the
directional derivative is -(‖rgrad_U‖²+‖grad_p‖²), with no additional factor of 2.
Reuse the current cost and accepted point without repeated matrix exponentials
or objective evaluations. NORM checks ΔJ≤cασ using its stable objective difference;
H reevaluates the same scalar objective with 256-bit arithmetic. Other objectives
use the scalar comparison. The step is successively halved. If no double-precision
step is accepted for NORM, repeat the sequence with higher-precision objective
evaluation. If neither sequence accepts a step, retain the final halved step
and continue the update.
"""
function _armijo_cached(cost_fn::Function, U::Matrix, params::Vector,
    rgrad_U::Matrix, grad_p::Vector, retract::Function, f0::Float64;
    α_init::Float64=1.0, c::Float64=1e-4, τ::Float64=0.5, max_iter::Int=50,
    trial_evaluator::Union{Nothing,Function}=nothing,
    precision_evaluator::Union{Nothing,Function}=nothing)
    evaluate_trial = if isnothing(trial_evaluator)
        (trial_U, trial_params, required_difference) -> begin
            trial_cost = cost_fn(trial_U, trial_params)
            return trial_cost, isfinite(trial_cost) && trial_cost <= f0+required_difference
        end
    else
        trial_evaluator(U, params, f0)
    end
    slope = -real(dot(rgrad_U, rgrad_U)) - real(dot(grad_p, grad_p))
    α = α_init
    for _ in 1:max_iter
        trial_U = retract(U, -α * rgrad_U)
        trial_params = params - α * grad_p
        trial_cost, accepted = evaluate_trial(trial_U, trial_params, c*α*slope)
        if accepted
            return (; accepted=true, α, U=trial_U, params=trial_params, cost=trial_cost)
        end
        α *= τ
    end
    if !isnothing(precision_evaluator)
        evaluate_precisely = precision_evaluator(U, params, f0)
        precise_alpha = α_init
        for _ in 1:max_iter
            new_U = retract(U, -precise_alpha * rgrad_U)
            new_params = params - precise_alpha * grad_p
            new_cost, accepted = evaluate_precisely(new_U, new_params, c*precise_alpha*slope)
            if accepted
                return (; accepted=true, α=precise_alpha, U=new_U, params=new_params, cost=new_cost)
            end
            precise_alpha *= τ
        end
    end
    # After 50 checks, retain the final halved step and continue the update.
    final_U = retract(U, -α * rgrad_U)
    final_params = params - α * grad_p
    return (; accepted=false, α, U=final_U, params=final_params,
        cost=cost_fn(final_U, final_params))
end

# Convenience interface returning only the step size.
function _armijo(cost_fn::Function, U::Matrix, params::Vector,
    rgrad_U::Matrix, grad_p::Vector, retract::Function; kwargs...)
    step = _armijo_cached(cost_fn, U, params, rgrad_U, grad_p, retract,
        cost_fn(U, params); kwargs...)
    return step.α
end

# Four-result analytic interface. Tangent projection belongs to the optimiser.
function _decoder_value_gradient(solver::UnitarySolver, U::Matrix, params::Vector)
    value, ambient_gradient, parameter_gradient = _decoder_ambient_value_gradient(solver, U, params)
    E = ambient_gradient.left * ambient_gradient.right
    return value, E, parameter_gradient, _rgrad_unitary(U, ambient_gradient)
end

