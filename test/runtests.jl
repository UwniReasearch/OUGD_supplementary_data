using Journal2Decoder, Test, LinearAlgebra, Random
const D = Journal2Decoder
BLAS.set_num_threads(1)

@testset "Single-bias paper configuration" begin
    solver = UnitaryDecoder(3)
    @test solver.num_params == 1
    initial = initialize_point(solver)
    @test length(initial.params) == 1
    X = solver.build_input(initial.params)
    @test all(X[4,:] .== only(initial.params))
    @test all(iszero,X[5:end,:])
end

@testset "Objective definitions and paper metrics" begin
    Q = [0.7 0.1; 0.3 0.9]
    @test D.compute_norm_cost_normalized(Q, 2) ≈ 0.1
    @test D.compute_extinction_ratio_cost(Q, 2) ≈ 1/0.7+1/0.9
    power = [0.6 0.1 0.1; 0.3 0.7 0.2; 0.1 0.2 0.7]
    metrics = paper_metrics(power)
    @test metrics.ER_dB[1] ≈ 10log10(0.6/0.3)
    @test metrics.CR_dB[1] ≈ 10log10(0.6/0.2)
end

@testset "Independent analytic and finite-difference derivatives" begin
    for target in (NORM, EXTINCTION)
        solver = UnitaryDecoder(2; cost_type=target, random_seed=42)
        initial = initialize_point(solver)
        value, E, gp = D._decoder_ambient_value_gradient(solver, initial.U, initial.params)
        cost(U,p) = solver.cost_fn(U*solver.build_input(p))
        Efd = D.numerical_gradient(U -> cost(U,initial.params), initial.U)
        pfd = D.numerical_gradient(p -> cost(initial.U,p), initial.params)
        @test value ≈ cost(initial.U,initial.params)
        @test norm(E.left*E.right-Efd)/norm(Efd) < 2e-6
        @test norm(gp-pfd)/norm(pfd) < 2e-6
    end
end

@testset "Small NORM and EXTINCTION solves" begin
    powers = Dict()
    for target in (NORM, EXTINCTION)
        solver = UnitaryDecoder(2; cost_type=target, random_seed=42)
        stats = OptimizationStats()
        result = optimize!(solver; verbose=false, gradient_backend=:analytic,
            max_iter=nothing, max_seconds=Inf, gtol=1e-8, stats)
        analysis = analyze_result(solver,result)
        power = normalized_power(analysis.Y)
        @test result.stop_reason == :gradient_tol
        @test gradient_norm(solver,result) < 1e-8
        @test norm(result.U'*result.U-I) < 1e-12
        powers[target] = power
    end
    @test maximum(abs.(powers[NORM]-powers[EXTINCTION])) < 1e-7
end

@testset "Three-bit CR objective decrease near convergence" begin
    solver = UnitaryDecoder(3,1;cost_type=EXTINCTION,random_seed=42)
    result = optimize!(solver;verbose=false,gradient_backend=:analytic,
        max_iter=nothing,max_seconds=Inf,gtol=1e-8)
    @test result.stop_reason == :gradient_tol
    @test gradient_norm(solver,result) < 1e-8
    @test norm(result.U'*result.U-I) < 1e-12
end

@testset "Published comparison methods preserve unitarity" begin
    solver = UnitaryDecoder(2; cost_type=NORM, random_seed=42)
    initial = initialize_point(solver)
    for method in (euclidean_gradient_descent, alternating_minimize)
        result,stats,_ = method(solver,initial.U,initial.params;gtol=1e-8)
        @test result.stop_reason == :gradient_tol
        @test gradient_norm(solver,result) < 1e-8
        @test norm(result.U'*result.U-I) < 1e-12
    end
end
