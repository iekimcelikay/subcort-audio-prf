# Subcortical Temporal pRF Modelling — Simulation Protocol

## 1. Forward Model Pipeline
The pipeline has two tiers. **Per-sequence stages (1–5)** process each unique stimulus sequence independently; their output is a pRF-weighted boxcar train at 1 ms resolution. **The run-assembly stage (6–7)** combines all per-sequence trains into a single full-run timeseries, applies AdapTrans *once* across the concatenated train, then convolves with the HRF.


```
╔══════════════════════════════════════════════════════╗
║  PER-SEQUENCE STAGES  (run once per unique sequence) ║
╚══════════════════════════════════════════════════════╝

Stimulus (tone sequence .npz)
        │
        ▼
Stage 1: Auditory Nerve (AN) Model
  → Pre-computed firing rates per CF × time
  → Stored as .npz; fixed input (no gradient)
        │
        ▼
Stage 2: Spectral Sharpening
  → r_sharp = r_AN ^ α
  → Free parameter: α
  → Mimics lateral inhibition (CN → IC)
        │
        ▼
Stages 3 & 5: Chunking & Boxcar Reconstruction
  → For each tone: compute mean firing rate over tone-ON window (+50 ms margin)
  → Reconstruct a continuous boxcar train:
       each boxcar has width = tone duration, amplitude = mean rate
  → Fixed computation; no free parameters
        │
        ▼
Stage 4: Duration Gaussian  [Models 2 and 4 only]
  → Scale each tone's boxcar amplitude by:
       G(d) = exp(-(d - pref_dur)² / (2 × σ_dur²))
  → Free parameters: pref_dur, σ_dur
        │
        ▼
  per-sequence pRF-weighted boxcar train  (1 ms resolution)

╔══════════════════════════════════════════════════════╗
║  RUN-ASSEMBLY STAGE  (assemble_run_bold, run_assembly.py) ║
╚══════════════════════════════════════════════════════╝

        │
        ▼
Stage 6a: Onset Placement
  → Place each per-sequence train at its run-level onset time
  → Sequences may repeat (counterbalanced design)
  → Null trials contribute zeros (silence)
  → Overlapping trains are summed (additive; back-to-back sequences)
  → Output: full_train  shape (n_1ms,)
        │
        ▼
Stage 6b: AdapTrans ON/OFF Filters  [Models 3 and 4 only]
  → Applied ONCE across the full assembled run (not per sequence)
  → Carry-over between consecutive sequences is modelled correctly:
       a sequence immediately following another inherits residual adaptation
  → ON channel: responds positively to onsets, negatively to offsets
  → OFF channel: responds positively to offsets, negatively to onsets
  → Sustained component (1 - w) passes through unchanged
  → Free parameters: w, ρ, τ_ON, τ_OFF
  → Output: on_response, off_response  shape (n_1ms,)
        │
        ▼
Stage 7: HRF Convolution  (hrf.py — SUBCORTICAL_PARAMS preset)
  → Convolve on_response and off_response separately with the HRF
  → Downsample to TR → bold_on, bold_off
  → bold_combined = ρ × bold_on + bold_off
  → HRF fixed in simulation (peak_delay=5 s, under_delay=9 s);
       fitted per voxel in real data
        │
        ▼
Predicted BOLD timecourse  →  Compare to measured BOLD  →  Loss
```
### Return keys from `assemble_run_bold`

| Key | Shape | Description |
|---|---|---|
| `full_train` | `(n_1ms,)` | Assembled pRF-weighted boxcar train, before AdapTrans |
| `on_response` | `(n_1ms,)` | AdapTrans ON channel (equals `full_train` when `apply_adaptrans_flag=False`) |
| `off_response` | `(n_1ms,)` | AdapTrans OFF channel (zeros when `apply_adaptrans_flag=False`) |
| `bold_on` | `(n_TR,)` | ON BOLD at TR resolution |
| `bold_off` | `(n_TR,)` | OFF BOLD at TR resolution |
| `bold_combined` | `(n_TR,)` | `ρ × bold_on + bold_off` — primary predicted timecourse |
| `t_tr` | `(n_TR,)` | Time axis in seconds |


## 2. Four-Model Hierarchy
Models are: 1) Spectral only(equivalent for visual spatial pRF) 2) Spectral + Duration 3) Spectral + Duration + AdapTrans 4) Spectral + AdapTrans
Parameters are **per voxel** — each voxel is characterised by its own set of values, analogous to how each voxel in visual pRF mapping has its own x, y, and σ.

---

### Model 1 — Spectral Only

> *Does the voxel have a preferred frequency? Can the AN model weighted by CF explain BOLD?*

**Stages active:** 1, 2, 3, 5 (per-seq) + 6a, 7 (run assembly)

| Parameter | Description | Range |
|---|---|---|
| CF index | Voxel's preferred frequency | 125–2500 Hz |
| α | Spectral sharpening exponent | TBD |

**Total free parameters per voxel: 2**

No explicit duration tuning; no onset/offset responses. The BOLD response to each frequency step is determined solely by how well the stimulus frequency matches the voxel's CF. This is the baseline model that we will build the temporal models upon. 

---

### Model 2 — Spectral + Duration

> *Does the voxel show selectivity for sound duration prefer tones of a specific duration?Are the neural timescales characterized by duration selectivity?*

**Stages active:** 1, 2, 3, 4, 5 (per-seq) + 6a, 7 (run assembly)

| Parameter | Description | Range |
|---|---|---|
| CF index | Voxel's preferred frequency | 125–2500 Hz |
| α | Spectral sharpening exponent | TBD |
| pref_dur | Voxel's preferred tone duration | 5-1000 ms |
| σ_dur | Width of duration tuning | TBD |

**Total free parameters per voxel: 4**

Duration tuning is explicit and parametric via a Gaussian. There are no onset/offset transient responses — the neural drive is purely sustained (boxcar shaped). This is the cleanest test of whether duration selectivity exists at all in the data, without any confound from adaptation dynamics.

---

### Model 3 - Spectral + AdapTrans (no Duration Gaussian)
> *Can we estimate temporal integration windows of subcortical auditory structures? Do the timescales arise from onset/offset dynamics?*

**Stages active:** 1, 2, 3, 5 (per-seq) + 6a, 6b, 7 (run assembly)

| Parameter | Description | Range |
|---|---|---|
| CF index | Voxel's preferred frequency | 125–2500 Hz |
| α | Spectral sharpening exponent | TBD |
| w | Sustained vs. transient balance | 0–1 |
| ρ | ON-to-OFF BOLD weighting ratio | > 0 |
| τ_ON | Time constant of onset filter | 5–1000 ms |
| τ_OFF | Time constant of offset filter | 5–1000 ms |

**Total free parameters per voxel: 6**

This model tests the hypothesis that timescales of IC and MGB are results of mechanistic adaptation dynamics and if there is duration tuning it is rather the overlap of ON and OFF reponses as a function of tone duration.

---

### Model 4 - Spectral + Duration + AdapTrans

> *Can onset/offset dynamics and explicit duration tuning together better explain the data?*
**Stages active:** 1, 2, 3, 4, 5 (per-seq) + 6a, 6b, 7 (run assembly)

| Parameter | Description | Range |
|---|---|---|
| CF index | Voxel's preferred frequency | 125–2500 Hz |
| α | Spectral sharpening exponent | TBD |
| pref_dur | Voxel's preferred tone duration | 5–1000 ms |
| σ_dur | Width of duration tuning | TBD |
| w | Sustained vs. transient balance (0=sustained, 1=transient) | 0–1 |
| ρ | ON-to-OFF BOLD weighting ratio | > 0 |
| τ_ON | Time constant of onset filter | 5–1000 ms * |
| τ_OFF | Time constant of offset filter | 5–1000 ms *  |

* We do not realistically expect time constants to go as low as 5 msec, these are just initial parameter ranges for simulations. 

**Total free parameters per voxel: 8**

This is the most complex model and the one requiring the most careful simulation validation. The key concern is **parameter coupling** between `pref_dur` and `τ_ON/τ_OFF`, because both mechanisms produce duration-dependent response profiles




