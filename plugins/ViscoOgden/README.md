# ViscoOgden: finite-strain viscoelastic Ogden for SOFA

This is a SOFA v26.06 plugin. It adds one component, `ViscoOgdenFEMForceField`, for
linear tetrahedra (one Gauss point, total Lagrangian).

SOFA's own `TetrahedronHyperelasticityFEMForceField` has a one-term Ogden law, but it is
purely elastic. This component adds:

- a multi-term Ogden law;
- viscoelasticity, with Prony-series internal variables Qᵢ. They carry the loading
  history, which the wall model in the thesis proposal needs.

## Model

```
W_iso = Σ_p 2 μ_p / α_p² (λ̄₁^α_p + λ̄₂^α_p + λ̄₃^α_p − 3)    λ̄_a = J^(−1/3) λ_a
U     = κ/2 (ln J)²
S     = S_vol + S_iso + Σ_i Q_i
Q_i^{n+1} = e^(−Δt/τ_i) Q_i^n + β_i e^(−Δt/2τ_i) (S_iso^{n+1} − S_iso^n)
```

The model and its update follow Simo (1987) and Holzapfel (2000, §6.10).

- **μ_p:** long-term (relaxed) moduli. The small-strain shear modulus is Σμ_p, following
  Budday et al. 2017.
- **Instantaneous moduli:** (1 + Σβᵢ) μ_p.
- **Tangent:** the consistent algorithmic tangent, in spectral form. Repeated eigenvalues use
  the limiting form. The stiffness matrix is assembled for direct solvers, through
  `buildStiffnessMatrix`.
- **When history updates:** the internal variables are committed once per time step, on
  `AnimateEndEvent`. They are never updated inside solver or LCP iterations. A reset clears
  them.

| Data | Meaning |
|---|---|
| `mu`, `alpha` | Ogden terms (equal lengths, α ≠ 0); `mu` in Pa |
| `bulkModulus` | κ [Pa] |
| `beta`, `tau` | Prony terms (β ≥ 0, τ > 0 s); leave empty for a hyperelastic material |
| `viscous` | `false` freezes the history (hyperelastic, long-term moduli) |
| `checkTangent` | logs analytic vs finite-difference element stiffness once (use with `printLog`) |
| `internalStressNorm` (out) | ‖Σ Qᵢ‖ per tetrahedron [Pa]: the internal state, e.g. for RL observations |
| `stress` (out) | total 2nd Piola–Kirchhoff stress per tetrahedron (flattened 3×3) |
| `elasticEnergy` (out) | long-term strain energy of the mesh [J] |

```python
root.addObject('RequiredPlugin', pluginName='ViscoOgden')
node.addObject('ViscoOgdenFEMForceField', mu=[1000], alpha=[-20], bulkModulus=5e4,
               beta=[2.0, 0.5], tau=[0.05, 1.0])
```

## Build

The plugin builds against the pixi SOFA in `~/Developer/sofa`:

```bash
cd ~/Developer/sofa
P=~/sofa/thesis/plugins/ViscoOgden; E=$PWD/.pixi/envs/supported-plugins
pixi run -e supported-plugins bash -c "cmake -S $P -B $P/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$E -DCMAKE_INSTALL_PREFIX=$E/plugins/ViscoOgden && ninja -C $P/build install"
grep -q '^ViscoOgden ' $E/lib/plugin_list.conf || printf '\nViscoOgden 0.1\n' >> $E/lib/plugin_list.conf
```

The install step puts the plugin in the SOFA env (`$E/plugins/ViscoOgden`). Listing it in
`$E/lib/plugin_list.conf` makes every runSofa / python-sofa from that env load it at start-up,
with no environment variable: SOFA.app, the Terminal, `rl/run.sh`.

- **After changing the C++:** re-run `ninja -C build install`. The env then gets the new
  version.
- **`run.sh`:** it sets `SOFA_PLUGIN_PATH=<this dir>/build`, so the tests use the fresh
  build even before installing.
- **If the env is reinstalled or SOFA rebuilt:** repeat both commands above. If
  `plugin_list.conf` is missing, copy it from `plugin_list.conf.default` first.

## Validation (`tests/`, all pass)

Run each test with `./run.sh tests/<test>.py`. `tests/plot_validation.py` draws
`tests/validation.png` from their results.

- **`homogeneous_test.py`:** prescribes a homogeneous deformation on a 5-tetrahedron cube and
  compares the stress with an independent NumPy reference. The reference stress comes from
  finite differences of W(C), followed by the same internal-variable update.
  - Cases: step-stretch relaxation, 1 Hz cyclic shear, and a two-term compressible Ogden.
  - Maximum relative error: 6e-10.
  - Hysteresis loop area: 529 Pa with viscosity, 4e-14 Pa with `viscous=0`.
- **`checkTangent`:** compares the analytic element stiffness with finite differences, both at
  rest (all eigenvalues equal) and deformed. Relative error: about 2e-10 to 5e-10.
- **`creep_test.py`:** a 10×10×20 mm column (768 tets) under 200 Pa, using implicit Euler with
  SparseLDL. The creep starts near the instantaneous hyperelastic bound, rises monotonically,
  and ends within 1.8 % of the long-term bound after 4 s.

## Demo

`examples/viscoogden_column.py` can be opened in SOFA.app. It loads the column for 1 s,
releases it for 2 s, and repeats. You can see the column creep under load and recover
slowly after release. The log shows the top displacement and the mean ‖Q‖.

## Not done yet

- Coupling the wall in the ventricle scene, which is currently a rigid surface.
- Parameter fitting to brain-tissue data.
- Multithreading.
- A slow, poro-like Prony term (large τ) is supported by the model, but has not been tuned.
