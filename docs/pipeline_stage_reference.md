# Forward Model Pipeline — Stage-by-Stage Reference

*May 2026*

This document describes the forward model pipeline for each of the four model variants. For each model, all active stages are listed with their data inputs, data outputs, and the brain process being modelled. A summary appears at the end.

Stages 1–5 are **per-sequence** (stateless, cacheable). Stages 6a–7 are **run-assembly** (context-dependent, applied once per run).

---

pipeline_stage_reference firing rate per tone-ON window (+50 ms margin); units: spk/s |
| **Brain** | Compression of within-tone dynamics into a single sustained drive value per tone. Models the assumption that downstream areas integrate the average AN rate over the tone window. Within-tone temporal structure is intentionally discarded here — it is irrecoverable from BOLD. |

### Stage 5 — Boxcar Reconstruction

| | |
|---|---|
| **Data in** | `n_tones` mean-rate scalars; tone onset and offset times (ms); total sequence duration (ms) |
| **Data out** | `train` — 1D array `(n_1ms_sequence,)` at 1 ms resolution; rectangular pulses, amplitude = mean rate, width = tone duration, zeros during ISIs |
| **Brain** | Re-embeds per-tone scalar responses into a continuous temporal signal — the sustained neural drive for this stimulus sequence. Amplitude encodes only spectral preference (via α). ISI zeros represent silence. |

### Stage 6a — Run Assembly

| | |
|---|---|
| **Data in** | Per-sequence `train` arrays; run design `[(seq_id, onset_s), ...]`; total run duration (s) |
| **Data out** | `full_train` — 1D array `(n_1ms_run,)` spanning the full fMRI run; per-sequence trains placed at their run-level onsets; null trials as zeros |
| **Brain** | The experimental design. Encodes the full temporal pattern of neural drive a voxel receives across one scanning run. Back-to-back sequences are concatenated; gaps are silence. |

### Stage 7 — HRF Convolution

| | |
|---|---|
| **Data in** | `full_train` `(n_1ms_run,)`; HRF kernel (`SUBCORTICAL_PARAMS`: peak 5 s, undershoot 9 s); TR |
| **Data out** | `bold_combined` `(n_TR,)` — predicted BOLD timecourse at TR resolution; `t_tr` `(n_TR,)` |
| **Brain** | Neurovascular coupling. Converts sustained neural drive into haemodynamic signal. The subcortical HRF peaks ~1 s earlier than the cortical canonical, reflecting faster subcortical neurovascular dynamics. |

---

## Model 2 — Spectral + Duration

**Free parameters:** CF index, α, pref_dur, σ_dur
**Stage sequence:** 1 → 2 → 3 → **4** → 5 → 6a → 7
**Differs from Model 1:** Stage 4 (Duration Gaussian) inserted between Chunking and Boxcar Reconstruction.

### Stage 1 — Auditory Nerve Model
*(identical to Model 1)*

### Stage 2 — Spectral Sharpening
*(identical to Model 1)*

### Stage 3 — Chunking
*(identical to Model 1)*

### Stage 4 — Duration Gaussian

| | |
|---|---|
| **Data in** | `n_tones` mean-rate scalars; `tone_dur_ms`; free parameters `pref_dur`, `σ_dur` |
| **Data out** | `n_tones` weighted scalars; each amplitude scaled by `G(d) = exp(-(d − pref_dur)² / 2σ_dur²)` |
| **Brain** | Duration-selective filtering in IC. IC neurons respond maximally to tones of a preferred duration and less to shorter or longer tones. Explicit, parametric model of duration tuning — a Gaussian tuning curve in duration space. Tones far from `pref_dur` contribute little neural drive regardless of their AN rate. |

### Stage 5 — Boxcar Reconstruction

| | |
|---|---|
| **Data in** | `n_tones` duration-weighted scalars; tone onset and offset times; total sequence duration |
| **Data out** | `train` — 1D array `(n_1ms_sequence,)` at 1 ms resolution; amplitude = duration-weighted rate |
| **Brain** | Same as Model 1, but boxcar amplitude now encodes both spectral preference (α) and duration preference (Gaussian). A voxel whose `pref_dur` does not match the stimulus duration produces low-amplitude boxcars. |

### Stage 6a — Run Assembly
*(identical to Model 1)*

### Stage 7 — HRF Convolution
*(identical to Model 1)*

---

## Model 3 — Spectral + Duration + AdapTrans

**Free parameters:** CF index, α, pref_dur, σ_dur, w, ON weight, OFF weight, τ_ON, τ_OFF
**Stage sequence:** 1 → 2 → 3 → **4** → 5 → 6a → **6b** → 7
**Differs from Model 2:** Stage 6b (AdapTrans) inserted between Run Assembly and HRF Convolution.

### Stage 1 — Auditory Nerve Model
*(identical to Model 1)*

### Stage 2 — Spectral Sharpening
*(identical to Model 1)*

### Stage 3 — Chunking
*(identical to Model 1)*

### Stage 4 — Duration Gaussian
*(identical to Model 2)*

### Stage 5 — Boxcar Reconstruction
*(identical to Model 2)*

### Stage 6a — Run Assembly
*(identical to Model 1)*

### Stage 6b — AdapTrans

| | |
|---|---|
| **Data in** | `full_train` `(n_1ms_run,)`; `CF_hz` → τ via Willmore equation `τ(f) = 500 − 105 × log₁₀(f)`; adaptation weight `w` |
| **Data out** | `on_response` `(n_1ms_run,)`, `off_response` `(n_1ms_run,)` |
| **Brain** | Onset- and offset-responding neurons in IC. These cells respond to changes in input rather than sustained levels. ON channel: burst at tone onset, suppressed by sustained drive, modulated by recent history via τ. OFF channel: burst at tone offset. Applied once across the full run so carry-over between sequences is preserved. Duration selectivity is partly explicit (Gaussian set amplitude in Stage 4) and partly implicit (ON/OFF overlap depends on tone duration relative to τ). |

### Stage 7 — HRF Convolution

| | |
|---|---|
| **Data in** | `on_response`, `off_response` `(n_1ms_run,)`; HRF kernel (`SUBCORTICAL_PARAMS`); TR; free parameter `ρ` |
| **Data out** | `bold_on` `(n_TR,)`, `bold_off` `(n_TR,)`, `bold_combined = ρ × bold_on + bold_off` `(n_TR,)` |
| **Brain** | Neurovascular coupling applied separately to ON and OFF channels. ON and OFF BOLD are combined with ratio ρ (free parameter per voxel): ρ > 1 → onset-dominated; ρ < 1 → offset-dominated; ρ = 1 → equal. |

---

## Model 4 — Spectral + AdapTrans

**Free parameters:** CF index, α, w, ON weight, OFF weight, τ_ON, τ_OFF
**Stage sequence:** 1 → 2 → 3 → 5 → 6a → **6b** → 7
**Differs from Model 3:** Stage 4 (Duration Gaussian) absent.
**Differs from Model 1:** Stage 6b (AdapTrans) present.

### Stage 1 — Auditory Nerve Model
*(identical to Model 1)*

### Stage 2 — Spectral Sharpening
*(identical to Model 1)*

### Stage 3 — Chunking
*(identical to Model 1)*

### Stage 5 — Boxcar Reconstruction

| | |
|---|---|
| **Data in** | `n_tones` unweighted mean-rate scalars; tone onset and offset times; total sequence duration |
| **Data out** | `train` — 1D array `(n_1ms_sequence,)` at 1 ms resolution; amplitude = raw sharpened mean AN rate, no duration weighting |
| **Brain** | Neural drive with no duration preference encoded. All tones of the same frequency within a sequence produce equal-amplitude boxcars. Duration selectivity is entirely absent at this stage — it emerges implicitly in Stage 6b from the relationship between τ and tone duration. |

### Stage 6a — Run Assembly
*(identical to Model 1)*

### Stage 6b — AdapTrans

| | |
|---|---|
| **Data in** | `full_train` `(n_1ms_run,)` with uniform-amplitude boxcars; `CF_hz`; adaptation weight `w` |
| **Data out** | `on_response` `(n_1ms_run,)`, `off_response` `(n_1ms_run,)` |
| **Brain** | Same IC onset/offset mechanism as Model 3, but AdapTrans carries all the duration selectivity. The time gap between ON and OFF responses equals tone duration. When tone duration >> τ: ON and OFF responses are separate peaks → strong combined response. When tone duration ≈ τ: ON response has not yet decayed when the tone ends → ON and OFF overlap and partially cancel → weaker response. A voxel implicitly prefers the duration where ON and OFF are most separated, determined by τ alone — no explicit tuning curve. |

### Stage 7 — HRF Convolution
*(identical to Model 3)*

---

## Summary

### Stage activity per model

| Stage | Description | Model 1 | Model 2 | Model 3 | Model 4 |
|---|---|:---:|:---:|:---:|:---:|
| 1 | Auditory Nerve Model | ✓ | ✓ | ✓ | ✓ |
| 2 | Spectral Sharpening | ✓ | ✓ | ✓ | ✓ |
| 3 | Chunking | ✓ | ✓ | ✓ | ✓ |
| 4 | Duration Gaussian | — | ✓ | ✓ | — |
| 5 | Boxcar Reconstruction | ✓ | ✓ | ✓ | ✓ |
| 6a | Run Assembly | ✓ | ✓ | ✓ | ✓ |
| 6b | AdapTrans | — | — | ✓ | ✓ |
| 7 | HRF Convolution | ✓ | ✓ | ✓ | ✓ |

### Where duration selectivity lives

| Model | Duration Gaussian (Stage 4) | AdapTrans (Stage 6b) | Mechanism |
|---|:---:|:---:|---|
| 1 | — | — | None — spectral response only |
| 2 | ✓ | — | Explicit — Gaussian amplitude weighting per tone |
| 3 | ✓ | ✓ | Both: explicit Gaussian + implicit τ/duration overlap |
| 4 | — | ✓ | Implicit only — τ/duration overlap geometry |

### Free parameters per model

| Model | Parameters | Count |
|---|---|:---:|
| 1 | CF, α | 2 |
| 2 | CF, α, pref_dur, σ_dur | 4 |
| 3 | CF, α, pref_dur, σ_dur, w, ρ, τ_ON, τ_OFF | 8 |
| 4 | CF, α, w, ρ, τ_ON, τ_OFF | 6 |

### Pipeline tiers

| Tier | Stages | Key property |
|---|---|---|
| Per-sequence | 1, 2, 3, 4, 5 | Stateless — same stimulus always gives same output; safe to pre-compute and cache per unique sequence |
| Run-assembly | 6a, 6b, 7 | Context-dependent — output depends on run design; AdapTrans (6b) additionally depends on stimulus history across the run |