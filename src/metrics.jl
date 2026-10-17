# Paper definitions. These functions do not use the reversed legacy helper labels.
normalized_power(Y::AbstractMatrix) = _normalized_power_columns(abs2.(Y))

"""Return ER and CR in dB, one value for each input word."""
function paper_metrics(power::AbstractMatrix{<:Real})
    n = size(power, 1)
    size(power, 2) == n || throw(DimensionMismatch("Expected one target per input word."))
    er, cr = zeros(n), zeros(n)
    for j in 1:n
        other = [power[i, j] for i in 1:n if i != j]
        er[j] = 10log10(power[j, j] / maximum(other))
        cr[j] = 10log10(power[j, j] / mean(other))
    end
    return (; ER_dB=er, CR_dB=cr)
end
