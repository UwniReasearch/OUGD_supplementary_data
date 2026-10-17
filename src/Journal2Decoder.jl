module Journal2Decoder

using LinearAlgebra, Random, Statistics

export NORM, EXTINCTION, UnitaryDecoder, OptimizationStats,
       initialize_point, optimize!, analyze_result, gradient_norm,
       normalized_power, paper_metrics, euclidean_gradient_descent, alternating_minimize

include("objectives.jl")
include("model.jl")
include("finite_difference_gradient.jl")
include("decoder_analytic.jl")
include("riemannian.jl")
include("solver.jl")
include("metrics.jl")
include("methods/unitary_coordinates.jl")
include("methods/euclidean_gradient_descent.jl")
include("methods/alternating_minimization.jl")

end
