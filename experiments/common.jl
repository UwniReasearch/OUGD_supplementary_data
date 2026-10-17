using Journal2Decoder
using LinearAlgebra, Random, Statistics, Printf
const D = Journal2Decoder
BLAS.set_num_threads(1)
const ROOT = normpath(joinpath(@__DIR__, ".."))

function option(name, default)
    position = findfirst(x -> startswith(x, name * "="), ARGS)
    return isnothing(position) ? default : split(ARGS[position], '='; limit=2)[2]
end

const OUT = abspath(option("--output-dir", joinpath(ROOT, "generated")))
mkpath(OUT)

function write_csv(path, rows)
    isempty(rows) && return
    fields = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(fields), ','))
        for row in rows
            println(io, join((getproperty(row, field) for field in fields), ','))
        end
    end
end

function read_csv(path)
    lines = readlines(path)
    fields = Tuple(Symbol.(split(first(lines), ',')))
    scalar(x) = x == "true" ? true : x == "false" ? false :
        something(tryparse(Int, x), tryparse(Float64, x), x)
    return [NamedTuple{fields}(Tuple(scalar.(split(line, ','))))
            for line in lines[2:end] if !isempty(line)]
end

# Recompute the returned numerical result without an analytical reference solution.
function verify(solver, result; gradient_backend::Symbol=:analytic)
    Y = D._decoder_output(solver, result.U, result.params)
    power = normalized_power(Y)
    J = D.compute_norm_cost_normalized(abs2.(Y), solver.n)
    H = D.compute_extinction_ratio_cost(abs2.(Y), solver.n)
    metrics = paper_metrics(power)
    return (; J, H,
        final_gradient_norm=gradient_norm(solver, result; gradient_backend),
        unitarity_residual=norm(result.U' * result.U - I),
        bias_norm=norm(result.params),
        ER_min_dB=minimum(metrics.ER_dB), CR_min_dB=minimum(metrics.CR_dB))
end

const _optimize = D._optimize
const _decoder_output = D._decoder_output
