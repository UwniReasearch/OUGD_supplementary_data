module EightLayerArchiveCheck

using TOML
include("verify_eight_layer_interconnect.jl")
const V = EightLayerVerification

function check_archive()
    directory = normpath(joinpath(@__DIR__, "..", "data", "interconnect_eight_layer"))
    stored = TOML.parsefile(joinpath(directory, "verification.toml"))
    report, summaries = V.verify_eight_layer_interconnect(directory; write_outputs=false)
    @assert stored["verification_passed"] && report["verification_passed"]
    for key in ("checks_run", "error_count", "measurement_rows", "component_readback_rows",
                "mzi_count", "mzi_columns", "mzi_column_counts")
        @assert report[key] == stored[key]
    end
    @assert report["comparison_tolerances"] == stored["comparison_tolerances"]

    ranges = [("input_power_W", :input_power_W_minmax, V.POWER_ATOL_W),
        ("total_output_W", :total_output_W_minmax, V.POWER_ATOL_W),
        ("throughput", :throughput_minmax, V.PARAMETER_ATOL),
        ("insertion_loss_dB", :insertion_loss_dB_minmax, V.DB_ATOL),
        ("ER_dB", :ER_dB_minmax, V.DB_ATOL),
        ("CR_dB", :CR_dB_minmax, V.DB_ATOL),
        ("target_output_W", :target_output_W_minmax, V.POWER_ATOL_W)]
    columns = ["condition", "runs", "target_largest_count"]
    for (name, _, _) in ranges
        append!(columns, [name*"_min", name*"_max"])
    end
    rows = V.read_csv(joinpath(directory, "summary.csv"), columns)
    @assert length(rows) == length(summaries) == 4
    @assert Set(row["condition"] for row in rows) == Set(V.CONDITIONS)
    for summary in summaries
        selected = filter(row -> row["condition"] == summary.condition, rows)
        @assert length(selected) == 1
        row = only(selected)
        @assert parse(Int, row["runs"]) == summary.runs
        @assert parse(Int, row["target_largest_count"]) == summary.target_largest_count
        for (name, property, atol) in ranges
            limits = getproperty(summary, property)
            @assert V.numeric_equal(parse(Float64, row[name*"_min"]), limits[1]; atol)
            @assert V.numeric_equal(parse(Float64, row[name*"_max"]), limits[2]; atol)
        end
    end
    println("Archived eight-layer numerical results and summary checked without writing files.")
    nothing
end

check_archive()
end
