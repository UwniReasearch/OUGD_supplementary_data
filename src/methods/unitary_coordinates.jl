# Mathematical coordinates for a unitary matrix and its common input bias.
# Loaded by Journal2Decoder. This file contains only the coordinate
# mapping and analytic chain rule; optimizer policy is in separate method files.
# The initial coordinates represent the supplied U0 and params0.
# The spectral divided-difference Frechet formula is a general matrix-function
# result, e.g. Higham & Relton (2014), https://doi.org/10.1137/130945259 .

using LinearAlgebra

"""Pack a Hermitian matrix in an orthonormal real Frobenius basis."""
function _unitary_coordinate_pack_hermitian(H::AbstractMatrix)
    n = size(H, 1)
    x = Vector{Float64}(undef, n*n)
    k = 0
    for i in 1:n
        k += 1
        x[k] = real(H[i, i])
    end
    for j in 2:n, i in 1:j-1
        k += 1
        x[k] = sqrt(2.0) * real(H[i, j])
        k += 1
        x[k] = sqrt(2.0) * imag(H[i, j])
    end
    return x
end

function _unitary_coordinate_unpack_hermitian(x::AbstractVector, n::Int)
    H = zeros(ComplexF64, n, n)
    k = 0
    for i in 1:n
        k += 1
        H[i, i] = x[k]
    end
    for j in 2:n, i in 1:j-1
        k += 1
        re = x[k] / sqrt(2.0)
        k += 1
        im = x[k] / sqrt(2.0)
        H[i, j] = complex(re, im)
        H[j, i] = conj(H[i, j])
    end
    return H
end

"""
General initialization U0 = exp(i H0) using its unitary Schur vectors and
principal eigenphases. It applies to any unitary matrix and uses no problem
solution. Mapping back to U0 is exact mathematically, up to floating-point error.
"""
function _unitary_initial_coordinates(U0::Matrix, params0::Vector)
    S = schur(U0)
    H0 = S.Z * Diagonal(angle.(diag(S.T))) * S.Z'
    h = _unitary_coordinate_pack_hermitian((H0 + H0') / 2)
    complex_params = !(eltype(params0) <: Real)
    return complex_params ? vcat(h, real.(params0), imag.(params0)) :
        vcat(h, Float64.(params0))
end

"""Decode real coordinates and cache the spectral data for the chain rule."""
function _unitary_coordinate_state(x::Vector{Float64}, n::Int, q::Int,
    complex_params::Bool)
    F = eigen(Hermitian(_unitary_coordinate_unpack_hermitian(x, n)))
    Q, λ = F.vectors, F.values
    U = (Q .* transpose(cis.(λ))) * Q'
    params = complex_params ? complex.(x[n*n+1:n*n+q], x[n*n+q+1:n*n+2q]) :
        copy(x[n*n+1:n*n+q])
    return (; U, params, Q, λ)
end

"""
Analytic Euclidean-coordinate gradient, plus the original (U,a) gradient norm.
For f(H)=exp(iH), D_ij = i exp(i(λ_i+λ_j)/2) sinc((λ_i-λ_j)/(2π)).
The adjoint is Q [conj(D) ⊙ (Q' E Q)] Q', Hermitian-projected because dH is
Hermitian. Julia sinc uses sin(πx)/(πx), including the exact x=0 limit.
"""
function _unitary_coordinate_value_gradient(solver::UnitarySolver, state,
    stats::Union{Nothing,OptimizationStats}=nothing)
    value, E, gp = _decoder_ambient_value_gradient(solver, state.U, state.params)
    if !isnothing(stats)
        stats.cost_evaluations += 1
        stats.gradient_evaluations += 1
    end
    rg = _rgrad_unitary(state.U, E)
    physical_gradient_norm = sqrt(real(dot(rg, rg)) + real(dot(gp, gp)))
    E_in_basis = (state.Q' * E.left) * (E.right * state.Q)
    λ = state.λ
    D = [im * cis((λ[i] + λ[j])/2) * sinc((λ[i] - λ[j])/(2π))
         for i in eachindex(λ), j in eachindex(λ)]
    raw = state.Q * (conj.(D) .* E_in_basis) * state.Q'
    gh = _unitary_coordinate_pack_hermitian((raw + raw') / 2)
    g = eltype(state.params) <: Real ? vcat(gh, Float64.(gp)) :
        vcat(gh, real.(gp), imag.(gp))
    return value, g, physical_gradient_norm
end

