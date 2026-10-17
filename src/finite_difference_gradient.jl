# Central finite differences, separate from analytic derivatives and optimisation.

"""
    numerical_gradient(cost_func, params, ε=1e-6) -> gradient

Compute the Euclidean gradient of a real-valued function by central differences.
The returned array has the same shape as the input. Differentiate real arrays
along each coordinate and complex arrays along both real and imaginary parts:
∂f/∂Re(z)+i∂f/∂Im(z)=2∂f/∂conj(z), with df=Re⟨gradient,dparams⟩.
The difference step ε does not change the objective or its domain.
"""
function numerical_gradient(cost_func::Function, params::AbstractArray{<:Complex}, ε::Float64=1e-6)
    gradient = zeros(ComplexF64, size(params))
    for index in eachindex(params)
        plus, minus = copy(params), copy(params)
        plus[index] += ε
        minus[index] -= ε
        real_partial = (cost_func(plus) - cost_func(minus)) / (2ε)
        plus[index], minus[index] = params[index] + im * ε, params[index] - im * ε
        imaginary_partial = (cost_func(plus) - cost_func(minus)) / (2ε)
        gradient[index] = real_partial + im * imaginary_partial
    end
    return gradient
end

function numerical_gradient(cost_func::Function, params::AbstractArray{<:Real}, ε::Float64=1e-6)
    gradient = zeros(Float64, size(params))
    for index in eachindex(params)
        plus, minus = copy(params), copy(params)
        plus[index] += ε
        minus[index] -= ε
        gradient[index] = (cost_func(plus) - cost_func(minus)) / (2ε)
    end
    return gradient
end

# Matrix and vector convenience interfaces.
numerical_gradient_matrix(cost_func::Function, matrix::Matrix, ε::Float64=1e-6) =
    numerical_gradient(cost_func, matrix, ε)
numerical_gradient_vector(cost_func::Function, vector::Vector, ε::Float64=1e-6) =
    numerical_gradient(cost_func, vector, ε)

"""
    _finite_difference_value_gradient(solver, U, params, cost_fn)
        -> (cost, ambient_U, gradient_params)

Return the objective, ambient matrix gradient and parameter gradient, using the
same interface as the analytic implementation. `cost_fn(U,params)` evaluates
the full real-valued objective. The matrix gradient is not projected onto the
manifold. With no free input parameters, the parameter gradient is empty.
All objective evaluations pass through `cost_fn` so that callers can count them.
"""
function _finite_difference_value_gradient(solver, U::Matrix, params::Vector, cost_fn::Function)
    value = cost_fn(U, params)
    ambient_gradient = numerical_gradient(V -> cost_fn(V, params), U)
    parameter_gradient = isempty(params) ? similar(params, 0) :
        numerical_gradient(p -> cost_fn(U, p), params)
    return value, ambient_gradient, parameter_gradient
end
