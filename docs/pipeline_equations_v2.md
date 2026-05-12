# Auditory pRF Pipeline — Mathematical Formulation (v2)

*May 2026 — Updated to reflect run-assembly architecture and ρ ON/OFF weighting.*
*See `pipeline_equations.md` for the previous version.*

This document describes the equations implemented across:
- `full_pipeline_with_adaptrans.py` — per-sequence stages (1–6)
- `run_assembly.py` — run-assembly stages (6a–8)
- `adaptrans_onoff_filters.py` — AdapTrans kernels
- `hrf.py` — HRF kernel and convolution

The pipeline has two tiers. **Per-sequence stages (1–6)** are stateless and cacheable — run once per unique stimulus sequence. **Run-assembly stages (6a–8)** are applied once per fMRI run and depend on the full run design.

### Model stage overview

| Stage | Description | M1 | M2 | M3 | M4 |
|---|---|:---:|:---:|:---:|:---:|
| 1–2 | AN model + spectral sharpening | ✓ | ✓ | ✓ | ✓ |
| 3 | Chunking | ✓ | ✓ | ✓ | ✓ |
| 4 | Duration Gaussian | — | ✓ | ✓ | — |
| 5–6 | Boxcar reconstruction | ✓ | ✓ | ✓ | ✓ |
| 6a | Run assembly | ✓ | ✓ | ✓ | ✓ |
| 7 | AdapTrans ON/OFF | — | — | ✓ | ✓ |
| 8 | HRF convolution | ✓ | ✓ | ✓ | ✓ |

---

## Notation

| Symbol | Description | Code variable |
|--------|-------------|---------------|
| $R(k, t)$ | Population PSTH: firing rate at CF index $k$, time $t$ | `population_psth[k, t]` |
| $\alpha$ | Power-law sharpening exponent | `alpha` |
| $k_0$ | Target CF index | `cf_index` |
| $f_{k_0}$ | Characteristic frequency of target CF (Hz) | `cf_hz` |
| $d$ | Tone duration (ms) | `tone_dur_ms` |
| $\hat{d}$ | Preferred duration (ms) | `pref_dur` |
| $\sigma_d$ | Duration tuning width (ms) | `sigma_dur` |
| $w$ | Adaptation weight (kernel shape) | `w` |
| $\tau(f)$ | CF-dependent time constant (ms) | `tau_ms` |
| $a$ | Exponential decay rate | `a` |
| $K$ | FIR kernel length (samples) | `K` |
| $S$ | Number of tones in the sequence | `n_tones` |
| $j$ | Stimulus presentation index in run | — |
| $t^{(j)}_\text{onset}$ | Run-level onset time of presentation $j$ (s) | `onset_s` in `run_design` |
| $x_\text{run}(n)$ | Full-run assembled neural train (1 ms resolution) | `full_train` |
| $\Delta t$ | Signal time step (s); default $10^{-3}$ s | `signal_dt_s` |
| $Q$ | Number of TRs in run | `n_trs` |
| $N$ | Samples per TR; $N = \text{round}(\text{TR}/\Delta t)$ | — |
| $B^\text{ON/OFF}[q]$ | Predicted BOLD ON/OFF at TR $q$ | `bold_on`, `bold_off` |
| $B[q]$ | Combined predicted BOLD timecourse | `bold_combined` |
| $\rho$ | ON-to-OFF BOLD weighting ratio | `rho` |
| $\delta_p, \sigma_p$ | HRF peak delay and dispersion (s) | `peak_delay`, `peak_disp` |
| $\delta_u, \sigma_u$ | HRF undershoot delay and dispersion (s) | `under_delay`, `under_disp` |
| $r$ | HRF peak-to-undershoot ratio | `p_u_ratio` |

---

## Step 1–2: Load and extract CF timecourse

Load the population PSTH matrix $R(k, t)$ of shape $(N_\text{CFs}, N_\text{bins})$ from the cochlear simulation `.npz` file. Select the target CF row:

$$R_{k_0}(t) = R(k_0, t)$$

---

## Step 3: Power-law sharpening (lateral inhibition)

Apply element-wise power-law to the **full population** matrix, then rescale to preserve the grand mean:

$$\tilde{R}(k, t) = R(k, t)^\alpha \cdot \frac{\langle R \rangle}{\langle R^\alpha \rangle}$$

where $\langle \cdot \rangle$ denotes the mean over all CFs and all time bins:

$$\langle R \rangle = \frac{1}{N_\text{CFs} \cdot N_\text{bins}} \sum_{k,t} R(k,t)$$

Then extract the target CF row from the sharpened population:

$$\tilde{R}_{k_0}(t) = \tilde{R}(k_0, t)$$

> **Why rescale?** The power-law amplifies differences between high and low rates (sharpening frequency tuning), but changes the overall magnitude. Dividing by the post-sharpening mean and multiplying by the pre-sharpening mean keeps the grand mean firing rate unchanged, so downstream stages see comparable amplitudes regardless of $\alpha$.

---

## Step 4: Chunk into tone-ON windows

Parse tone timing from the stimulus filename (`dur<N>ms`, `isi<N>ms`). Compute onset/offset times for each tone $s = 1, \ldots, S$:

$$t_s^\text{on} = (s-1) \cdot (d + \Delta_\text{ISI})$$

$$t_s^\text{off} = t_s^\text{on} + d$$

Extract each tone-ON window (plus a 50 ms margin) from $\tilde{R}_{k_0}(t)$ and compute the mean firing rate per tone:

$$\bar{r}_s = \frac{1}{|\mathcal{W}_s|} \sum_{t \in \mathcal{W}_s} \tilde{R}_{k_0}(t)$$

where $\mathcal{W}_s = \{t : t_s^\text{on} \leq t < t_s^\text{off} + 50\text{ ms}\}$.

---

## Step 5: Duration Gaussian filter [Models 2 and 3 only]

Weight each tone's mean rate by a Gaussian centred on the preferred duration:

$$g(d) = \frac{1}{\sqrt{2\pi}\,\sigma_d} \exp\!\left(-\frac{(d - \hat{d})^2}{2\sigma_d^2}\right)$$

$$p_s = \bar{r}_s \cdot g(d)$$

where $p_s$ is the **pRF response scalar** for tone $s$. Since all tones in a pure-tone sequence share the same duration $d$, $g(d)$ is a single scalar — it modulates the overall amplitude based on how well the tone duration matches the neuron's preferred duration.

For **Models 1 and 4** (no Duration Gaussian): $p_s = \bar{r}_s$.

---

## Step 6: Build boxcar impulse train

Construct a 1 ms resolution signal where each tone's interval is filled with its pRF response amplitude:

$$x^{(j)}(n) = \sum_{s=1}^{S} p_s \cdot \mathbf{1}_{[n_s^\text{on},\, n_s^\text{off})}(n)$$

where $n_s^\text{on} = \text{round}(t_s^\text{on} / \Delta t)$ and $n_s^\text{off} = \text{round}(t_s^\text{off} / \Delta t)$.

**Per-sequence output:** `train`, shape $(n_\text{1ms,seq},)$. Cached per unique sequence; reused at every onset where the sequence appears.

---

## Step 6a: Run Assembly

Place each per-sequence boxcar train at its run-level onset, forming the full-run neural drive signal:

$$x_\text{run}(n) = \sum_{j} x^{(j)}\!\left(n - n^{(j)}_\text{onset}\right), \quad n^{(j)}_\text{onset} = \text{round}\!\left(\frac{t^{(j)}_\text{onset}}{\Delta t}\right)$$

where $j$ indexes all stimulus presentations in the run design. Null trials contribute zero. Sequences may repeat at different onsets (counterbalanced design). Overlapping trains are summed (additive).

The signal is padded with zeros at the start (`pad_value=0.0`) — the run begins from silence, ensuring the first onset is not suppressed by a spurious adaptation state.

**Code:** `assemble_run_bold` in `run_assembly.py`. Output: `full_train`, shape $(n_\text{1ms,run},)$.

---

## Step 7: AdapTrans ON/OFF filters [Models 3 and 4 only]

AdapTrans is applied **once** to the full-run signal $x_\text{run}(n)$. A single causal convolution preserves adaptation state across tone boundaries and across sequence boundaries (carry-over suppression).

### Time constant (Willmore et al., 2016, rescaled for subcortex)

$$\tau(f) = 0.15 \cdot \left(500 - 105 \cdot \log_{10}(f)\right) \quad \text{[ms]}$$

### Decay rate

$$a = e^{-\Delta t_\text{ms} / \tau(f)}$$

where $\Delta t_\text{ms} = \Delta t \times 1000$.

### Kernel length (auto)

If not specified:

$$K = \left\lceil 3 \cdot \tau_\text{max} / \Delta t_\text{ms} \right\rceil$$

### ON kernel (FIR, length $K$)

$$C = \frac{1}{\displaystyle\sum_{j=0}^{K-2} a^j}$$

$$h_\text{ON}[n] = \begin{cases} +1 & n = 0 \quad \text{(current sample)} \\ -C \cdot w \cdot a^{n-1} & n = 1, \ldots, K-1 \quad \text{(exponentially weighted past)} \end{cases}$$

The ON kernel computes: *current sample minus adapted running average of the past*. Large positive output = onset (increase from baseline).

### OFF kernel (FIR, length $K$)

The OFF kernel detects *decreases* — it compares the exponentially weighted past against the (discounted) current sample:

$$h_\text{OFF}[n] = \begin{cases} -w & n = 0 \quad \text{(current sample, discounted by } w\text{)} \\ +C \cdot a^{n-1} & n = 1, \ldots, K-1 \quad \text{(exponentially weighted past)} \end{cases}$$

### ON/OFF kernel asymmetry

The OFF kernel is **not** the exact negative of the ON kernel. This is intentional. Comparing the two side by side:

| Tap | $h_\text{ON}[n]$ | $h_\text{OFF}[n]$ |
|-----|-------------------|--------------------|
| $n = 0$ (present) | $+1$ | $-w$ |
| $n \geq 1$ (past) | $-C w a^{n-1}$ | $+C a^{n-1}$ |

The asymmetry: **ON discounts the past by $w$; OFF discounts the present by $w$.**

Intuitively:
- **ON**: "Is the current sample ($\times 1$) larger than the adapted past ($\times w$)?"
- **OFF**: "Is the adapted past ($\times 1$) larger than the current sample ($\times w$)?"

When $w = 1$, the asymmetry vanishes and $h_\text{OFF} = -h_\text{ON}$ exactly.

### Code shortcut for OFF kernel derivation

```python
off_kernel = -on_kernel / w      # fixes taps 1..K-1, but tap 0 = -1/w (wrong)
off_kernel[0] = -w               # overwrite tap 0 to correct value
```

### Full-run convolution

$$Y^\text{ON}(n) = \sum_{m=0}^{K-1} h_\text{ON}[m] \cdot x_\text{run}(n - m)$$

$$Y^\text{OFF}(n) = \sum_{m=0}^{K-1} h_\text{OFF}[m] \cdot x_\text{run}(n - m)$$

This is a single causal convolution — **not** per-tone isolation followed by superposition. The adaptation state at any sample $n$ reflects the full history of $x_\text{run}$ up to that point, including contributions from preceding sequences in the run. This correctly models carry-over suppression between back-to-back stimuli.

**Code:** `apply_adaptrans` in `adaptrans_onoff_filters.py`. Outputs: `on_response`, `off_response`, each shape $(n_\text{1ms,run},)$.

For **Models 1 and 2** (no AdapTrans): $Y^\text{ON} = x_\text{run}$, $Y^\text{OFF} = \mathbf{0}$.

---

## Step 8: HRF Convolution

### HRF kernel

A normalised double-gamma kernel (`SUBCORTICAL_PARAMS`: $\delta_p = 5$ s, $\sigma_p = 1$ s, $\delta_u = 9$ s, $\sigma_u = 1$ s, $r = 6$):

$$h_\text{HRF}(t) = \text{Gamma}\!\left(t;\, \frac{\delta_p}{\sigma_p},\, \sigma_p\right) - \frac{1}{r}\,\text{Gamma}\!\left(t;\, \frac{\delta_u}{\sigma_u},\, \sigma_u\right)$$

Normalised so the positive lobe has unit area:

$$\sum_{n \geq 0} \max\!\left(h_\text{HRF}[n], 0\right) \cdot \Delta t = 1$$

**Code:** `build_hrf_kernel` in `hrf.py`.

### Causal convolution at 1 ms resolution

$$\hat{B}^\text{ON}(n) = \left(\sum_{m=0}^{M-1} h_\text{HRF}[m] \cdot Y^\text{ON}(n-m)\right) \cdot \Delta t$$

$$\hat{B}^\text{OFF}(n) = \left(\sum_{m=0}^{M-1} h_\text{HRF}[m] \cdot Y^\text{OFF}(n-m)\right) \cdot \Delta t$$

where $M$ is the kernel length in samples.

### Downsampling to TR

$$B^\text{ON}[q] = \hat{B}^\text{ON}(q \cdot N), \quad B^\text{OFF}[q] = \hat{B}^\text{OFF}(q \cdot N), \quad N = \text{round}\!\left(\frac{\text{TR}}{\Delta t}\right)$$

### ON/OFF combination

$$B[q] = \rho \cdot B^\text{ON}[q] + B^\text{OFF}[q]$$

where $\rho$ is a free parameter per voxel:
- $\rho > 1$: onset-dominated BOLD response
- $\rho = 1$: equal ON/OFF contribution (default)
- $\rho < 1$: offset-dominated BOLD response

**Code:** `convolve_hrf` in `hrf.py`; combination in `assemble_run_bold` in `run_assembly.py`.

---

## Full model (compact form)

For parameter set $\Theta = \{k_0, \alpha, [\hat{d}, \sigma_d,] w, \rho\}$ (brackets = Models 2 & 3 only):

$$B[q] = \rho \cdot \bigl(h_\text{HRF} * Y^\text{ON}\bigr)[q \cdot N] \;+\; \bigl(h_\text{HRF} * Y^\text{OFF}\bigr)[q \cdot N]$$

where:

$$Y^\text{ON/OFF} = h_\text{ON/OFF} * x_\text{run} \quad \text{[Models 3 \& 4]}$$
$$Y^\text{ON} = x_\text{run},\; Y^\text{OFF} = \mathbf{0} \quad \text{[Models 1 \& 2]}$$

and the full-run signal is assembled from per-sequence boxcars:

$$x_\text{run}(n) = \sum_j x^{(j)}\!\left(n - n_\text{onset}^{(j)}\right)$$

$$x^{(j)}(n) = \sum_{s=1}^{S} p_s \cdot \mathbf{1}_{[n_s^\text{on},\, n_s^\text{off})}(n)$$

$$p_s = \begin{cases} \bar{r}_s \cdot g(d) & \text{Models 2 \& 3 (Duration Gaussian active)} \\ \bar{r}_s & \text{Models 1 \& 4 (no Duration Gaussian)} \end{cases}$$