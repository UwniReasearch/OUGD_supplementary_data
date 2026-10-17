include("common.jl")

# Export revised single-bias Frobenius results in input-symbol power units.
# These new power tables are not the source data for the original Figure 3.
rows = NamedTuple[]
power_rows = NamedTuple[]
for m in 2:6
    solver = UnitaryDecoder(m, 1; cost_type=NORM, random_seed=42)
    result = optimize!(solver; verbose=false, gradient_backend=:analytic,
        max_iter=nothing, max_seconds=Inf, gtol=1e-8)
    power = abs2.(D._decoder_output(solver, result.U, result.params))
    fractions = normalized_power(D._decoder_output(solver, result.U, result.params))
    append!(power_rows,
        [(; bits=m, output_port=i, input_word=j, power=power[i,j],
            normalized_power=fractions[i,j])
            for j in 1:solver.n for i in 1:solver.n])
    write_csv(joinpath(OUT, "decoder_power.csv"), power_rows)
    push!(rows, (; bits=m, num_bias_params=1, seed=42,
        max_iter="none", max_seconds=Inf, requested_gtol=1e-8,
        stop_reason=string(result.stop_reason), verify(solver,result)...))
end
write_csv(joinpath(OUT,"decoder_power_summary.csv"), rows)
println("Exported five single-bias power maps.")
