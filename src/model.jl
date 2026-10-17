# Problem definition and the original objective option names.

@enum CostType begin
    NORM = 1
    EXTINCTION = 2
end

# ============================================================================
# General solver for unitary transformations
# ============================================================================

"""
    UnitarySolver{T<:Number}

Define the optimisation problem through two functions:
- `build_input(params::Vector{T}) -> Matrix{ComplexF64}` builds the input matrix X.
- `cost_fn(Y::Matrix{ComplexF64}) -> Float64` evaluates the objective at Y = U * X.

`T` is the parameter type, either ComplexF64 or Float64.
When `num_params == 0`, the input is fixed and only U is optimised.

Use `UnitaryDecoder(num_bits; ...)` to construct the decoder model.
"""
mutable struct UnitarySolver{T<:Number}
    n::Int                              # Unitary matrix dimension: n by n
    num_params::Int                     # Number of input parameters; zero fixes the input
    build_input::Function               # Vector{T} -> Matrix{ComplexF64}
    cost_fn::Function                   # Matrix{ComplexF64} -> Float64
    label::String
    random_seed::Int
    last_result::Union{Nothing,Any}
    decoder_config::Union{Nothing,NamedTuple}  # BPSK decoder configuration
end

"""
    UnitarySolver(n, build_input, cost_fn, T; num_params, label, random_seed)

Construct a solver from an input builder and an objective function.
"""
function UnitarySolver(
    n::Int,
    build_input::Function,
    cost_fn::Function,
    ::Type{T}=ComplexF64;
    num_params::Int=0,
    label::String="Unitary - $(n)",
    random_seed::Int=42,
) where {T<:Number}
    return UnitarySolver{T}(n, num_params, build_input, cost_fn, label, random_seed, nothing, nothing)
end

"""
    UnitarySolver(n, X, cost_fn; ...)

Construct a solver with a fixed input matrix and `num_params=0`.
"""
function UnitarySolver(n::Int, X::Matrix, cost_fn::Function; kwargs...)
    X_fixed = convert(Matrix{ComplexF64}, X)
    return UnitarySolver(n, _ -> X_fixed, cost_fn; num_params=0, kwargs...)
end

# ============================================================================
# Decoder construction
# ============================================================================

"""
    build_bpsk_patterns_generic(num_bits) -> Matrix{ComplexF64}

Return a num_bits by 2^num_bits matrix with one BPSK input word per column.
"""
function build_bpsk_patterns_generic(num_bits::Int)::Matrix{ComplexF64}
    n = 2^num_bits
    patterns = zeros(ComplexF64, num_bits, n)
    for state in 0:n-1
        bits = digits(state, base=2, pad=num_bits)
        patterns[:, state+1] = [b == 0 ? 1.0 : -1.0 for b in reverse(bits)]
    end
    return patterns
end

"""
    UnitaryDecoder(num_bits, num_free_params, T; cost_type, random_seed) -> UnitarySolver

Construct a decoder with BPSK symbol rows, shared bias rows and unused zero rows.
Select the objective through `cost_type`: NORM or EXTINCTION.
"""
function UnitaryDecoder(
    num_bits::Int,
    num_free_params::Int,
    ::Type{T};
    cost_type::CostType=NORM,
    random_seed::Int=42,
) where {T<:Number}
    n = 2^num_bits
    bpsk = build_bpsk_patterns_generic(num_bits)
    rest = n - num_bits - num_free_params

    build_input = function(params::Vector)
        [
            bpsk;
            [p for p in params, _ in 1:n];
            zeros(ComplexF64, rest, n)
        ]
    end

    cost_fn = let cost_type=cost_type, n=n
        (Y::Matrix) -> begin
            P = compute_power(Y)
            if cost_type == NORM
                return compute_norm_cost_normalized(P, n)
            else
                return compute_extinction_ratio_cost(P, n)
            end
        end
    end

    solver = UnitarySolver(n, build_input, cost_fn, T;
        num_params=num_free_params,
        label="Decoder - $(n)",
        random_seed=random_seed,
    )
    solver.decoder_config = (; num_bits, bpsk, cost_type)
    return solver
end

UnitaryDecoder(num_bits::Int, num_free_params::Int; kwargs...) =
    UnitaryDecoder(num_bits, num_free_params, ComplexF64; kwargs...)

UnitaryDecoder(num_bits::Int; kwargs...) =
    UnitaryDecoder(num_bits, 1; kwargs...)

