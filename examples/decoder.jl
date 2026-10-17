using Journal2Decoder
using LinearAlgebra, Printf
BLAS.set_num_threads(1)

# Three-bit NORM example. The objective-comparison script tests both costs at two to six bits.
solver = UnitaryDecoder(3, 1; cost_type=NORM, random_seed=42)
stats = OptimizationStats()
result = optimize!(solver; verbose=false, gradient_backend=:analytic,
    max_iter=nothing, max_seconds=Inf, gtol=1e-8, stats)
analysis = analyze_result(solver, result)
power = normalized_power(analysis.Y)
ratios = paper_metrics(power)

@printf("Objective: %.12g\n", analysis.cost)
@printf("Gradient norm: %.6g\n", gradient_norm(solver, result))
println("Stop reason: ", result.stop_reason)
println("Updates: ", stats.iterations)
@printf("Unitarity residual: %.6g\n", norm(result.U' * result.U - I))
@printf("Minimum ER: %.8f dB; minimum CR: %.8f dB\n",
    minimum(ratios.ER_dB), minimum(ratios.CR_dB))
println("Normalised output power, port by input word:")
show(stdout, "text/plain", round.(power; digits=8))
println()
