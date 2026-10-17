#!/usr/bin/env julia
module EightLayerVerification

using Printf
using TOML

export verify_eight_layer_interconnect

const CONDITIONS = ["ideal_reference","balanced_limit","cross_low_limit","cross_high_limit"]
const INPUT_FILES = ["decoder_results.csv","model_parameters.csv","component_readback.csv",
                     "mesh_phases_before.csv","mesh_phases_after.csv"]

# These are comparison tolerances for exported numbers, not simulation or
# optimisation stopping criteria. Power is measured in W and angle in rad.
const NUMBER_RTOL = 1e-10
const POWER_ATOL_W = 1e-15
const DB_ATOL = 1e-8
const PARAMETER_ATOL = 1e-12
const PHASE_ATOL_RAD = 1e-12

# The native export contains only unquoted scalar fields.
csv_fields(line) = String.(strip.(split(line,',';keepempty=true)))

function read_csv(path,required_columns)
    lines = readlines(path)
    isempty(lines) && error("Empty input file: $path")
    header = csv_fields(replace(first(lines),'\ufeff'=>""))
    length(unique(header))==length(header) || error("Duplicate CSV headers: $path")
    all(key in header for key in required_columns) ||
        error("Missing required columns in $path: $(setdiff(required_columns,header))")
    rows = Dict{String,String}[]
    for (i,line) in enumerate(lines[2:end])
        isempty(strip(line)) && continue
        fields = csv_fields(line)
        length(fields)==length(header) || error("Wrong field count in $path at line $(i+1)")
        push!(rows,Dict(header .=> fields))
    end
    rows
end

number(row,key) = parse(Float64,row[key])
function boolean(row,key)
    value = lowercase(row[key])
    value in ("true","false") && return value=="true"
    numeric = parse(Float64,value)
    numeric in (0.0,1.0) || error("$key is not a Boolean flag: $value")
    numeric==1.0
end

function numeric_equal(a,b;atol=PARAMETER_ATOL,rtol=NUMBER_RTOL)
    isfinite(a) && isfinite(b) && return isapprox(a,b;atol=atol,rtol=rtol)
    isinf(a) && a==b
end

function scenario_parameters(condition)
    if condition=="ideal_reference"
        return (excess=0.0,cross=0.5,waveguide_loss=0.0)
    elseif condition=="balanced_limit"
        return (excess=4-10log10(2),cross=0.5,waveguide_loss=150.0)
    end
    weak,strong = 10.0^(-4/10),10.0^(-3/10)
    total = weak+strong
    condition in ("cross_low_limit","cross_high_limit") || error("Unknown condition: $condition")
    fraction = condition=="cross_low_limit" ? weak/total : strong/total
    (excess=-10log10(total),cross=fraction,waveguide_loss=150.0)
end

"""
    verify_eight_layer_interconnect(directory; write_outputs=true)

Independently recalculate the 32 optical results and inspect every exported
coupler, phase and output-waveguide readback. Non-target-largest observations
remain valid experimental outcomes and are counted rather than rejected.

Set `write_outputs=false` to inspect an archived dataset without changing it.
"""
function verify_eight_layer_interconnect(directory;write_outputs=true)
    missing = [name for name in INPUT_FILES if !isfile(joinpath(directory,name))]
    isempty(missing) || error("Missing eight-layer measurement data in $directory: $(join(missing,", ")). No verification outputs were created.")
    results = read_csv(joinpath(directory,"decoder_results.csv"),
        vcat(["condition","input_word","input_power_W","total_output_W","throughput",
              "insertion_loss_dB","ER_dB","CR_dB","target_is_largest"],
             ["output_$(i)_W" for i in 1:8]))
    parameters = read_csv(joinpath(directory,"model_parameters.csv"),
        ["condition","coupler_excess_loss_dB","coupler_TE_cross_power_fraction",
         "output_waveguide_length_m","waveguide_loss_dB_per_m","detector_noise_enabled",
         "detector_dark_current_A","detector_saturation_enabled","laser_power_W",
         "laser_frequency_Hz","sample_rate_Hz","time_window_s","bias_phase_rad"])
    components = read_csv(joinpath(directory,"component_readback.csv"),
        ["condition","element","property","value"])
    phase_columns = ["name","column","row","theta_rad","phi_rad","internal_phase_1_rad","internal_phase_2_rad"]
    before = read_csv(joinpath(directory,"mesh_phases_before.csv"),phase_columns)
    after = read_csv(joinpath(directory,"mesh_phases_after.csv"),phase_columns)
    errors = String[]
    checks_run = Ref(0)
    residuals = Dict{String,Float64}()
    function check(ok,message)
        checks_run[] += 1
        ok || push!(errors,message)
        ok
    end
    function check_number(actual,expected,label;atol=PARAMETER_ATOL,rtol=NUMBER_RTOL,residual_key="")
        if !isempty(residual_key) && isfinite(actual) && isfinite(expected)
            residuals[residual_key] = max(get(residuals,residual_key,0.0),abs(actual-expected))
        end
        check(numeric_equal(actual,expected;atol=atol,rtol=rtol),"$label: $actual; expected $expected")
    end
    function condition_set(rows,label)
        check(Set(r["condition"] for r in rows)==Set(CONDITIONS),"$label condition set differs from the four required conditions")
    end
    check(length(results)==32,"Expected 32 optical measurement rows; found $(length(results))")
    check(length(parameters)==4,"Expected four model-parameter rows; found $(length(parameters))")
    check(length(components)==736,"Expected 736 component-property readbacks; found $(length(components))")
    condition_set(results,"Measurement")
    condition_set(parameters,"Model parameter")
    condition_set(components,"Component readback")
    check(length(before)==length(after)==28,"The phase snapshots must each contain 28 MZIs")
    before_map = Dict(r["name"]=>r for r in before)
    after_map = Dict(r["name"]=>r for r in after)
    check(length(before_map)==28 && length(after_map)==28,"MZI names are not unique")
    check(Set(keys(before_map))==Set(keys(after_map)),"MZI names changed between phase snapshots")
    coordinates = [(parse(Int,r["column"]),parse(Int,r["row"])) for r in before]
    check(length(unique(coordinates))==28,"MZI column/row positions are not unique")
    check(Set(first.(coordinates))==Set(1:8),"The circuit does not contain all eight MZI columns")
    column_counts = [count(p->p[1]==column,coordinates) for column in 1:8]
    check(column_counts==[4,3,4,3,4,3,4,3] || column_counts==[3,4,3,4,3,4,3,4],
        "The eight MZI columns must alternate between four and three MZIs")
    for (name,row) in before_map
        haskey(after_map,name) || continue
        other = after_map[name]
        check(row["column"]==other["column"] && row["row"]==other["row"],"MZI $name moved in the phase snapshots")
        for key in phase_columns[4:end]
            check_number(number(other,key),number(row,key),"MZI $name unchanged $key";
                atol=PHASE_ATOL_RAD,rtol=0.0,residual_key="phase_snapshot_max_error_rad")
        end
    end
    parameter_map = Dict(r["condition"]=>r for r in parameters)
    check(length(parameter_map)==4,"Model-parameter conditions are not unique")
    reference = get(parameter_map,"ideal_reference",isempty(parameters) ? nothing : first(parameters))
    for condition in CONDITIONS
        haskey(parameter_map,condition) || continue
        row = parameter_map[condition]
        expected = scenario_parameters(condition)
        for (key,value) in (("coupler_excess_loss_dB",expected.excess),
            ("coupler_TE_cross_power_fraction",expected.cross),("output_waveguide_length_m",0.002),
            ("waveguide_loss_dB_per_m",expected.waveguide_loss),("laser_power_W",0.001),
            ("detector_dark_current_A",0.0))
            check_number(number(row,key),value,"$condition model $key")
        end
        check(!boolean(row,"detector_noise_enabled"),"$condition detector noise is enabled")
        check(!boolean(row,"detector_saturation_enabled"),"$condition detector saturation is enabled")
        for key in ("laser_frequency_Hz","sample_rate_Hz","time_window_s")
            check(number(row,key)>0 && isfinite(number(row,key)),"$condition $key must be finite and positive")
        end
        if reference!==nothing
            for key in ("laser_frequency_Hz","sample_rate_Hz","time_window_s","bias_phase_rad")
                check_number(number(row,key),number(reference,key),"$condition fixed $key";
                    atol=key=="bias_phase_rad" ? PHASE_ATOL_RAD : PARAMETER_ATOL,
                    rtol=key=="bias_phase_rad" ? 0.0 : NUMBER_RTOL)
            end
        end
    end

    expected_couplers = Set(name*"::C_"*string(i) for name in keys(before_map) for i in 1:2)
    expected_mzis = Set(keys(before_map))
    expected_waveguides = Set("OUTPUT_WG_"*string(i) for i in 1:8)
    property_elements = Dict("insertion_loss_dB"=>expected_couplers,
        "TE_cross_power_fraction"=>expected_couplers,"theta_rad"=>expected_mzis,
        "phi_rad"=>expected_mzis,"loss_dB_per_m"=>expected_waveguides,"length_m"=>expected_waveguides)
    for condition in CONDITIONS
        rows = filter(r->r["condition"]==condition,components)
        pairs = [(r["element"],r["property"]) for r in rows]
        check(length(unique(pairs))==length(pairs),"$condition has duplicate component/property readbacks")
        check(Set(r["property"] for r in rows)==Set(keys(property_elements)),"$condition has missing or unexpected component properties")
        for (property,expected_names) in property_elements
            selected = filter(r->r["property"]==property,rows)
            check(length(selected)==length(expected_names),"$condition $property readback count is incorrect")
            check(Set(r["element"] for r in selected)==expected_names,"$condition $property element names are incorrect")
        end
        haskey(parameter_map,condition) || continue
        model = parameter_map[condition]
        for row in rows
            name,property = row["element"],row["property"]
            actual = number(row,"value")
            expected = if property=="insertion_loss_dB"
                number(model,"coupler_excess_loss_dB")
            elseif property=="TE_cross_power_fraction"
                number(model,"coupler_TE_cross_power_fraction")
            elseif property=="loss_dB_per_m"
                number(model,"waveguide_loss_dB_per_m")
            elseif property=="length_m"
                number(model,"output_waveguide_length_m")
            elseif property in ("theta_rad","phi_rad") && haskey(before_map,name)
                number(before_map[name],property)
            else
                continue
            end
            check_number(actual,expected,"$condition $name $property";
                atol=property in ("theta_rad","phi_rad") ? PHASE_ATOL_RAD : PARAMETER_ATOL,
                rtol=property in ("theta_rad","phi_rad") ? 0.0 : NUMBER_RTOL,
                residual_key="component_"*property*"_max_error")
        end
    end

    derived = NamedTuple[]
    expected_words = Set(string(i;base=2,pad=3) for i in 0:7)
    for condition in CONDITIONS
        rows = filter(r->r["condition"]==condition,results)
        check(length(rows)==8,"$condition must have eight input words")
        check(Set(r["input_word"] for r in rows)==expected_words,"$condition must contain each three-bit word exactly once")
    end
    for (index,row) in enumerate(results)
        condition,word = row["condition"],row["input_word"]
        word in expected_words || (check(false,"Invalid three-bit input word at measurement row $index");continue)
        target = parse(Int,word;base=2)+1
        powers = [number(row,"output_$(i)_W") for i in 1:8]
        check(all(isfinite,powers) && all(>=(0.0),powers),"$condition/$word contains non-finite or negative output power")
        input = number(row,"input_power_W")
        check(isfinite(input) && input>0,"$condition/$word input power must be finite and positive")
        haskey(parameter_map,condition) && check_number(input,number(parameter_map[condition],"laser_power_W"),
            "$condition/$word input power including bias";atol=POWER_ATOL_W,residual_key="input_power_max_error_W")
        total = sum(powers)
        check(total>0,"$condition/$word total output power must be positive")
        throughput = total/input
        loss = -10log10(throughput)
        off_target = powers[setdiff(1:8,[target])]
        er = 10log10(powers[target]/maximum(off_target))
        cr = 10log10(powers[target]/(sum(off_target)/7))
        largest = powers[target]>maximum(off_target)
        check_number(number(row,"total_output_W"),total,"$condition/$word total_output_W";
            atol=POWER_ATOL_W,residual_key="total_output_max_error_W")
        check_number(number(row,"throughput"),throughput,"$condition/$word throughput";
            residual_key="throughput_max_error")
        for (key,value) in (("insertion_loss_dB",loss),("ER_dB",er),("CR_dB",cr))
            check_number(number(row,key),value,"$condition/$word $key";
                atol=DB_ATOL,residual_key=key*"_max_error")
        end
        check(boolean(row,"target_is_largest")==largest,"$condition/$word reported target-largest flag is inconsistent")
        check(throughput<=1+NUMBER_RTOL,"$condition/$word output energy exceeds the stated input")
        push!(derived,(condition=condition,word=word,input=input,total=total,throughput=throughput,
            loss=loss,ER=er,CR=cr,target_power=powers[target],target_largest=largest))
    end
    summaries = NamedTuple[]
    for condition in CONDITIONS
        rows = filter(r->r.condition==condition,derived)
        isempty(rows) && continue
        limits(key) = extrema(getproperty(r,key) for r in rows)
        push!(summaries,(condition=condition,runs=length(rows),
            input_power_W_minmax=limits(:input),total_output_W_minmax=limits(:total),
            throughput_minmax=limits(:throughput),insertion_loss_dB_minmax=limits(:loss),
            ER_dB_minmax=limits(:ER),CR_dB_minmax=limits(:CR),
            target_output_W_minmax=limits(:target_power),target_largest_count=count(r->r.target_largest,rows)))
    end
    report = Dict("verification_passed"=>isempty(errors),"checks_run"=>checks_run[],
        "error_count"=>length(errors),"errors"=>errors,
        "measurement_rows"=>length(results),"component_readback_rows"=>length(components),
        "mzi_count"=>length(before),"mzi_columns"=>length(unique(first.(coordinates))),
        "mzi_column_counts"=>column_counts,
        "maximum_residuals"=>residuals,
        "comparison_tolerances"=>Dict("numeric_relative"=>NUMBER_RTOL,"power_absolute_W"=>POWER_ATOL_W,
            "dB_absolute"=>DB_ATOL,"parameter_absolute"=>PARAMETER_ATOL,"phase_absolute_rad"=>PHASE_ATOL_RAD),
        "scope"=>"Eight-layer 28-MZI original-circuit measurements; four deterministic optical conditions and eight input words per condition.",
        "interpretation"=>["Non-ideal conditions use conservative limits of strict component bounds; these are not measured or guaranteed device parameters.",
            "Each 2 mm output waveguide is a scenario assumption; no drawing-derived internal routing length is used.",
            "Throughput includes the bias input in the stated 1 mW total input power.",
            "ER uses the largest non-target output; CR uses the mean of all seven non-target outputs.",
            "Target-largest means strictly greater than every non-target output; counts are outcomes and are not required to be eight.",
            "Noise and detector flags are checked from model_parameters.csv; individual detector properties are outside the component readback table.",
            "All fixed MZI phases are checked against both snapshots and every condition readback."])
    if write_outputs
        open(joinpath(directory,"summary.csv"),"w") do io
            println(io,"condition,runs,input_power_W_min,input_power_W_max,total_output_W_min,total_output_W_max,throughput_min,throughput_max,insertion_loss_dB_min,insertion_loss_dB_max,ER_dB_min,ER_dB_max,CR_dB_min,CR_dB_max,target_output_W_min,target_output_W_max,target_largest_count")
            for r in summaries
                values = [r.input_power_W_minmax...,r.total_output_W_minmax...,r.throughput_minmax...,
                          r.insertion_loss_dB_minmax...,r.ER_dB_minmax...,r.CR_dB_minmax...,r.target_output_W_minmax...]
                println(io,r.condition,',',r.runs,',',join([@sprintf("%.15g",x) for x in values],','),',',r.target_largest_count)
            end
        end
        open(joinpath(directory,"verification.toml"),"w") do io
            TOML.print(io,report;sorted=true)
        end
    end
    println("Eight-layer optical verification: ",isempty(errors) ? "PASS" : "FAIL",
            " (",checks_run[]," checks; ",length(errors)," errors)")
    for message in errors
        println(stderr,"  ",message)
    end
    report,summaries
end

end # module

if abspath(PROGRAM_FILE)==@__FILE__
    args = copy(ARGS)
    write_outputs = "--write-outputs" in args
    filter!(arg -> arg != "--check-only" && arg != "--write-outputs",args)
    length(args)<=1 || error("Usage: julia verify_eight_layer_interconnect.jl [data_directory] [--write-outputs]")
    default_directory = joinpath(dirname(@__DIR__),"data","interconnect_eight_layer")
    directory = isempty(args) ? default_directory : abspath(only(args))
    report,_ = EightLayerVerification.verify_eight_layer_interconnect(directory;write_outputs)
    report["verification_passed"] || exit(1)
end
