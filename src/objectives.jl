# Scalar objectives and stable evaluation of objective changes.

compute_power(Y::Matrix{ComplexF64}) = abs2.(Y)

function compute_extinction_ratio_cost(P_pred::Matrix{Float64}, num_patterns::Int)::Float64
    return _cr_cost_from_power(P_pred, num_patterns)
end

# The same CR-based objective also supports precise step comparisons.
function _cr_cost_from_power(P_pred::AbstractMatrix{<:Real}, num_patterns::Int)
    total_cost = zero(eltype(P_pred))
    for j in 1:num_patterns
        target_power = P_pred[j, j]
        total_power = _column_total_power(P_pred, j)
        total_cost += total_power / target_power
    end
    return total_cost
end

"""
    _column_total_power(P, column) -> total

Return the finite, strictly positive total power of a column. Zero or nonfinite
column power is outside the normalisation domain. This domain check introduces
neither a noise threshold nor an alternative objective.
"""
function _column_total_power(P::AbstractMatrix{<:Real}, column::Int)
    total = sum(view(P, :, column))
    isfinite(total) && total > 0 ||
        throw(DomainError(total, "Column $column must have finite, strictly positive total power for normalisation."))
    return total
end

# Shared column normalisation and objective summation.
function _normalized_power_columns(P::AbstractMatrix{<:Real})
    normalized = copy(P)
    for j in axes(P, 2)
        view(normalized, :, j) ./= _column_total_power(P, j)
    end
    return normalized
end

function _norm_cost_from_normalized_power(normalized_power::AbstractMatrix{<:Real}, num_patterns::Int)
    cost = zero(eltype(normalized_power))
    for j in 1:num_patterns
        residual = copy(view(normalized_power, :, j))
        residual[j] -= 1.0
        cost += 0.5 * sum(abs2, residual)
    end
    return cost
end

"""
    compute_norm_cost_normalized(P_pred, num_patterns)

Evaluate the normalised Frobenius objective ½Σⱼ‖P[:,j]/sⱼ-eⱼ‖²,
where sⱼ is the total power of column j. Each total must be finite and strictly
positive; otherwise a DomainError is raised. In the unitary BPSK model,
sⱼ=m+‖params‖²≥m>0, so no power threshold is required.
"""
function compute_norm_cost_normalized(P_pred::Matrix{Float64}, num_patterns::Int)::Float64
    return _norm_cost_from_normalized_power(_normalized_power_columns(P_pred), num_patterns)
end

"""
    compute_norm_cost_difference(current_normalized_power, trial_normalized_power) -> Float64

Evaluate the change in the same normalised Frobenius objective J=½‖Q-I‖²:
ΔJ=Σᵢⱼ(Qtrial-Qcurrent)ᵢⱼ((Qtrial+Qcurrent)/2-I)ᵢⱼ.
Both arguments are column-normalised power tables. This algebraic identity
does not require unitarity and preserves the objective and Armijo condition.
It avoids subtracting two separately accumulated scalar costs. No acceptance
tolerance is added.
"""
function compute_norm_cost_difference(current_normalized_power::Matrix{Float64},
    trial_normalized_power::Matrix{Float64})::Float64
    difference = trial_normalized_power-current_normalized_power
    average_residual = (trial_normalized_power+current_normalized_power)/2
    for j in axes(average_residual, 2)
        average_residual[j, j] -= 1.0
    end
    return sum(difference .* average_residual)
end

