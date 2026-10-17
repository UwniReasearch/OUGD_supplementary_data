# supplementary data and source

Manuscript: Design of Arbitrary Optical Unitary Gates via Riemannian Gradient
Descent: Application to Optical Decoders

This self-contained Julia package contains the decoder implementation and the
numerical tables used in the revised manuscript. Julia 1.12.7 was used for the
archived experiments. Dependencies are Julia's bundled standard libraries and
Statistics 1.11.5, pinned in Manifest.toml. First-time installation may download
Statistics and Julia library artifacts; the calculations themselves need no network.
Archived timings were measured on an Apple M2 with one Julia thread and one BLAS thread.

From this directory:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. test/runtests.jl
julia --startup-file=no --project=. examples/decoder.jl
julia --startup-file=no --project=. verification/check_results.jl
julia --startup-file=no --project=. verification/check_interconnect_eight_layer.jl
```

`examples/decoder.jl` exposes the bit count, objective and seed directly. It
prints the objective, gradient norm, update count, unitarity residual and output
power table for the three-bit NORM example. Every paper experiment explicitly
uses `UnitaryDecoder(m, 1)`, with one complex bias parameter and zero input at
the remaining unused ports. The default paper constructor also selects one bias.

## Definitions and source locations

For output field `Y=U*X`, let `p=abs2.(Y)`, `s[j]=sum(p[:,j])` and
`Q[i,j]=p[i,j]/s[j]`. Rows are output ports; columns are input words in ascending
binary order. Column `j` has target port `j`. The first m input rows contain the
BPSK words. Row m+1 contains the common complex bias. The remaining n-m-1 rows
are zero. The parameter space used by the paper is `U(n) × C¹`.
With unrestricted U in U(n), multiple bias inputs shared by all input words are
equivalent to one bias input. A unitary change of basis in the bias subspace
preserves the attainable output fields.

| Item | Definition or source |
| --- | --- |
| `NORM` | `J=0.5*sum(abs2,Q-I)` in `src/objectives.jl` |
| `EXTINCTION` | `H=sum(1/Q[j,j])` in `src/objectives.jl` |
| Analytic derivative | `src/decoder_analytic.jl` |
| Central finite difference | `src/finite_difference_gradient.jl` |
| Riemannian gradient descent | `src/solver.jl`, `src/riemannian.jl` |
| Euclidean gradient descent | `src/methods/euclidean_gradient_descent.jl` |
| Hermitian coordinates `U=exp(iH)` | `src/methods/unitary_coordinates.jl` |
| Alternating minimisation | `src/methods/alternating_minimization.jl` |
| ER | Target power divided by largest non-target power, reported in dB |
| CR | Target power divided by mean non-target power, reported in dB |

Manuscript Table 1 reports arithmetic means of the per-input ER and CR values
in decibels. The `ER_min_dB` and `CR_min_dB` fields in the reproduction CSV files
instead report minima over input words, as their names indicate.

`EXTINCTION` is a retained implementation option name for the paper's CR-based
objective `H`. It sums reciprocal target-power fractions. For the linear CR
ratios `C[j]`, `H=n+(n-1)*sum(1/C[j])`, so minimising it maximises their harmonic
mean. It does not directly maximise the worst-case ER. The unreported legacy
`CONTRAST` option is omitted from this paper-specific package. `paper_metrics`
computes ER and CR directly from the definitions above, avoiding reversed legacy
helper labels.

The analytic derivative uses `dJ=real(dot(G,dY))`, with `G=2Y.*(dJ/dp)`.
For the ambient matrix gradient `G_U`, the unitary component is
`U*(U'*G_U-G_U'*U)/2`. The analytic and finite-difference
implementations are separate. The optimizer selects one before its loop.

The solver preserves the measured Armijo rule: initial step 1, coefficient
`1e-4`, halving, 50 double-precision step checks, then the same sequence with
256-bit objective evaluation for `NORM` if needed. The stable objective
difference is an algebraic identity. It does not change the objective or
stopping threshold. If neither sequence accepts a step, the original final
`2^-50` update is retained. The optimisation loops have no time or iteration
limit when `max_iter=nothing,max_seconds=Inf` is specified. The 50 step checks
are not an optimisation iteration limit.

All solvers use random QR initialisation and one complex Gaussian bias sample.
The recorded `num_bias_params` field must be one. Returned results are checked by
recomputing their objective, product-manifold gradient and unitarity residual.

## Reproduce the numerical experiments

All new output goes to `generated/`; archived `data/` is never overwritten.
The experiment scripts set one BLAS thread. Use one Julia thread as below to
match the archived measurements. Compilation is excluded from formal scaling
and method timings; runtimes depend on the machine and Julia version.

```sh
julia --startup-file=no --threads=1 --project=. experiments/objective_comparison.jl --bits=2,3,4,5,6
julia --startup-file=no --threads=1 --project=. experiments/random_initializations.jl --runs=100
julia --startup-file=no --threads=1 --project=. experiments/scaling.jl --max-bits=9
julia --startup-file=no --threads=1 --project=. experiments/power_maps.jl
julia --startup-file=no --threads=1 --project=. experiments/method_comparison.jl --bits=2,3,4,5,6,7,8,9 --repeats=3
julia --startup-file=no --threads=1 --project=. experiments/method_comparison.jl --bits=3 --seeds=1008,2015,3022,4029,5036,6043,7050,8057,9064,10071 --repeats=1 --prefix=method_initializations
```

Random initialisations use `seed=i*1000+(i*7%999)+1`. They save the initial
objective and every update, as well as the final recomputed objective and
gradient. Scaling and method comparisons terminate at the original product
gradient norm below `1e-8`, without time or iteration caps. AM block tolerances
are `1e-8/sqrt(2)`; each block solve is a numerical first-order solve. EGD uses
ordinary gradient descent in fixed Hermitian coordinates. It preserves
unitarity and uses no momentum or quasi-Newton update. Method timing includes
initialisation and the EGD coordinate conversion. Scaling timing excludes
initialisation. The iteration units of RGD, EGD and AM differ; elapsed time and
actual objective and gradient counts are the comparison quantities. The archived
method key `rgd_joint` denotes the paper's Riemannian gradient descent. It is
retained to match the original data and does not name a separate method.

The objective entry point uses analytic gradients with no iteration or time
limit. The command above reproduces the five archived pairs from seed 42;
`--bits` and `--seed` select other cases. All ten runs reach the gradient threshold.
For H, the Armijo condition uses 256-bit reevaluation of the same objective;
the gradient and matrix updates remain in double precision. This avoids
loss of the objective decrease through scalar rounding near convergence.
The independent numerical derivative is used only by
derivative checks, not by the paper's timing comparisons.

## Archived data

`data/` contains the single-bias 100-start summary and complete trajectories,
2–9 bit scaling, 72 method timing rows and their summaries, 30 paired method
starts, the objective-comparison results, and output powers from the revised
single-bias Frobenius reproduction experiments. The manuscript retains the
original Figure 3; these new power tables are not its original source data.
`decoder_power.csv` gives both absolute power in input-symbol power units and
column-normalised power for two to six bits, identified by its `bits` column. The corresponding
`decoder_power_summary.csv` records the model, stopping criterion and residuals.
`scaling_power.csv` retains the normalised power matrices exported after the timed
solves for two to nine bits, identified by its `bits` column.
The tables retain the recorded empirical fields used in this paper. Timings,
update counts, costs, gradients and measured port powers are unchanged.
`NaN` in method subcounts means unrecorded or inapplicable. It does not mean
zero work. Summary timing values are
medians and ranges of the recorded repetitions, not complexity proofs.

The INTERCONNECT data are grouped into two independent studies. Neither study
is a measurement of fabricated devices.

- `data/interconnect/` retains the earlier 24 static simulations with exploratory
  loss and detector-noise assumptions. Its two original CSV files are unchanged.
  These results are separate from the optical sensitivity study below.
- `data/interconnect_eight_layer/` contains 32 static optical simulations of the
  original eight-layer circuit with 28 MZIs and 56 couplers. The eight MZI columns
  alternate between four and three MZIs. The same programmed MZI phases and bias
  phase are retained for all eight input words in each of four conditions. There
  is no phase reoptimisation in this study.

The four optical conditions use the following specified parameters. Exact
readbacks are in `model_parameters.csv` and `component_readback.csv`.

| Condition | Coupler excess loss, $L_{\mathrm{ex}}/\mathrm{dB}$ | Coupler cross-power fraction | Output-waveguide attenuation, $\alpha_{\mathrm{wg}}/(\mathrm{dB}\,\mathrm{m}^{-1})$ |
| --- | ---: | ---: | ---: |
| `ideal_reference` | 0 | 0.5 | 0 |
| `balanced_limit` | 0.989700043360188 | 0.5 | 150 |
| `cross_low_limit` | 0.460981089561328 | 0.442688366237707 | 150 |
| `cross_high_limit` | 0.460981089561328 | 0.557311633762293 | 150 |

These are sensitivity scenarios with conservative component limits, not measured
or guaranteed device parameters. Each output waveguide has an assumed length of
2 mm. Internal routing lengths are not inferred from a layout. Total source power
is 1 mW, including the common bias. Detector noise, dark current and saturation
are disabled in all four optical conditions. These static optical powers do not
establish BER, receiver sensitivity or electrical energy consumption.

The new dataset provides the five native CSV exports, `summary.csv` and
`verification.toml`. The phase snapshots before and after the study are identical.
The standalone verifier checks all 32 results, all 736 component-property
readbacks, the eight-layer arrangement, fixed phases and recorded parameters.
It recomputes total power, throughput, insertion loss, ER and CR from the eight
port powers. A target-largest count is checked against the actual powers rather
than imposed as a pass condition. The wrapper also checks the archived summary.
Both public verification commands above
read this dataset without modifying its files. The numerical checks are retained
from the original verifier, with its standalone default path adapted to the data
directory inside this package. The standalone verifier is also read-only by
default; `--write-outputs` regenerates its summary and verification report.
No INTERCONNECT installation is needed to verify the exported results.

Native circuit projects, simulation scripts and private design documents are not
included in this Julia package.

The installed source needs
no external project paths, network connection, credentials or plotting package
to execute. Plot data are provided as ordinary CSV tables.
