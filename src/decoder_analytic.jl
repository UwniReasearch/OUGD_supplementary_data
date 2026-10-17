# BPSK objectives and analytic derivatives. The optimiser and step selection are separate.

"""
    _decoder_output(solver, U, params) -> Y

The first m input rows contain all BPSK words. The next q rows contain shared
bias amplitudes in params. Compute
Y = U[:,1:m] B + (U[:,m+1:m+q] params) 1ᵀ
without forming the full input matrix. The input rank is at most m+1.
"""
function _decoder_output(solver::UnitarySolver, U::Matrix, params::Vector)
    cfg = solver.decoder_config
    isnothing(cfg) && throw(ArgumentError("Analytic decoder functions require UnitaryDecoder."))
    m = cfg.num_bits
    Y = U[:, 1:m] * cfg.bpsk
    if !isempty(params)
        reference_field = U[:, m+1:m+length(params)] * params
        Y .+= reference_field
    end
    return Y
end

"""
    _decoder_cost_output_gradient(Y, cost_type) -> (cost, gradient_Y)

Return the real-valued objective and complex Euclidean output gradient, with
df = Re⟨gradient_Y,dY⟩. For P=|Y|² and D=∂f/∂P, gradient_Y=2Y⊙D.
The factor 2 follows from d|Y|²=2Re(conj(Y)dY).

NORM is ½Σⱼ‖P[:,j]/sⱼ-eⱼ‖², with sⱼ=ΣᵢPᵢⱼ. Differentiating the normalisation gives
D[:,j]=(qⱼ-eⱼ-⟨qⱼ-eⱼ,qⱼ⟩1)/sⱼ, where qⱼ=P[:,j]/sⱼ.
Each column total must be finite and strictly positive. In the unitary BPSK model,
sⱼ=m+‖params‖²≥m>0; zero-power columns are outside the objective domain.

EXTINCTION is the retained option name for the CR-based objective Σⱼsⱼ/Pⱼⱼ.
Its gradient requires strictly positive target powers Pⱼⱼ. This analytic interface
supports NORM and EXTINCTION; the separate finite-difference implementation
provides independent derivative checks.
"""
function _decoder_cost_output_gradient(Y::Matrix, cost_type::CostType)
    cost_type in (NORM, EXTINCTION) ||
        throw(ArgumentError("Analytic differentiation supports NORM and EXTINCTION."))
    n = size(Y, 2)
    P = abs2.(Y)
    power_gradient = zeros(Float64, size(P))
    value = 0.0
    for j in 1:n
        power = view(P, :, j)
        gradient_column = view(power_gradient, :, j)
        total = sum(power)
        target = power[j]
        if cost_type == NORM
            total = _column_total_power(P, j)
            normalized_power = power ./ total
            residual = copy(normalized_power)
            residual[j] -= 1.0
            value += 0.5 * sum(abs2, residual)
            weighted_residual = dot(residual, normalized_power)
            gradient_column .= (residual .- weighted_residual) ./ total
        elseif cost_type == EXTINCTION
            total = _column_total_power(P, j)
            isfinite(target) && target > 0 ||
                throw(DomainError(target, "The EXTINCTION gradient requires finite, strictly positive target power in column $j."))
            value += total / target
            gradient_column .= 1.0 / target
            gradient_column[j] = (1.0 - total / target) / target
        else
            throw(ArgumentError("Unsupported decoder objective: $cost_type"))
        end
    end
    return value, 2 .* Y .* power_gradient
end

"""
    _decoder_ambient_value_gradient(solver, U, params)
        -> (cost, ambient_U, gradient_params)

Return analytic derivatives using the same interface as finite differences,
without manifold projection or optimisation. For G=∇Yf, the matrix gradient is
∇Uf=G Xᴴ and the bias gradient is Urefᴴ G1. Real parameters receive real gradients;
complex parameters use the convention df=Re⟨g,dp⟩.

ambient_U=(left=L,right=C) represents the ambient gradient E=L*C, with
L=[G Bᴴ, G1]. The first m rows of C select the symbol columns; its last row
places paramsᴴ in the bias columns. This factorisation retains rank at most
m+1 and avoids a dense gradient product. With no bias parameters, L=G Bᴴ
and C selects only the symbol columns.
"""
function _decoder_ambient_value_gradient(solver::UnitarySolver, U::Matrix, params::Vector)
    cfg = solver.decoder_config
    Y = _decoder_output(solver, U, params)
    value, output_gradient = _decoder_cost_output_gradient(Y, cfg.cost_type)
    m, q = cfg.num_bits, length(params)
    signal_gradient = output_gradient * cfg.bpsk'
    rank = m + (q > 0)
    input_factor = zeros(ComplexF64, rank, size(U, 2))
    for i in 1:m
        input_factor[i, i] = 1.0
    end
    if q > 0
        summed_output_gradient = vec(sum(output_gradient; dims=2))
        input_factor[end, m+1:m+q] .= conj.(params)
        left_factor = hcat(signal_gradient, summed_output_gradient)
        parameter_gradient = U[:, m+1:m+q]' * summed_output_gradient
        eltype(params) <: Real && (parameter_gradient = real.(parameter_gradient))
    else
        left_factor = signal_gradient
        parameter_gradient = similar(params, 0)
    end
    return value, (; left=left_factor, right=input_factor), parameter_gradient
end
