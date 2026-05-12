# subcort-audio-prf
Standalone repository with only the necessary code for Gari

- cochlea model source code not included.
(read `doc_install_cochlea.md`)
- project root defined locally, needs to be fixed.
- imports use the old package structure, needs to be fixed

1) First you need to create sound sequences
    python stimuli/save_sequences_greenwood_automated.py
2) then $PROJECT_ROOT$/cochlea_waw_psth_runner.py

3) Then with the output of that we proceed with the prf-pipeline,
 however the current full_pipeline script works only on one sound sequence(one trial), and it doesn't yet implement 'run_assembly'. This part is in progress.

There's a test script that can show the workflow: 'subcort-audio-prf/auditory_prf/tests/test_run_assembly.py'

Probably nothing works right now, because the scripts are moved from a different repository and paths are not correct. But this doesn't make them illegible.

| File                                         | Role                                                                                      |
| -------------------------------------------- | ----------------------------------------------------------------------------------------- |
| prf_pipeline/full_pipeline_with_adaptrans.py | Main per-sequence pipeline entry point (Currently based on Model4 (Duration + AdapTrans)) |
| prf_pipeline/run_assembly.py                 | Run-level assembly + HRF                                                                  |
| prf_pipeline/adaptrans_onoff_filters.py      | AdapTrans ON/OFF kernels                                                                  |
| prf_pipeline/chunk_timecourse.py             | Chunking stage                                                                            |
| prf_pipeline/duration_models.py              | Duration Gaussian                                                                         |
| prf_pipeline/powerlaw_function.py            | Spectral sharpening                                                                       |
| prf_pipeline/load_extract_cf_timecourse.py   | Loads .npz, extracts CF row                                                               |
| prf_pipeline/hrf.py                          | HRF kernel + convolution (NumPy)                                                          |

## Flowchart of forward model pipeline

``` mermaid
---
config:
  layout: dagre
  theme: redux
---
flowchart TB
 subgraph S1["Per-sequence stages (stateless, cacheable)"]
    direction TB
        IN1["Acoustic waveform \n (Sequence of tones with silence in between)"]
        A["`**Stage 1**
          Auditory Nerve Model`"]
        N1["`Population PSTH
          (n_CFs × n_time_bins)`"]
        B["`**Stage 2**
          Spectral Sharpening`"]
        N2["Sharpened 1D timecourse\n(target CF)"]
        C["`**Stage 3**
           Chunking`"]
        N3["Mean firing rate per tone"]
        D["`**Stage 4**
          Duration Gaussian`"]
        N4["Duration-weighted scalars"]
        E["`**Stage 5**
          Boxcar Reconstruction`"]
        N5["Sequence train\n(1D, 1 ms resolution)"]
  end
 subgraph S2["Run-assembly stages (context-dependent)"]
    direction TB
        F["`**Stage 6a**
          Run Assembly`"]
        N6["Full run train\n(1D)"]
        G["`**Stage 6b**
          AdapTrans`"]
        N7["ON/OFF responses"]
        H["`**Stage 7**
          HRF Convolution`"]
        I["Synthetic noiseless BOLD\n(TR resolution)"]
  end
 subgraph L["Legend"]
    direction TB
        L1["Per-sequence Stage"]
        L2["Run-assembly Stage"]
        L3["Input/Output"]
        L4["Final Output"]
        FP["Free parameter"]
  end
    IN1 --> A
    A --> N1
    N1 --> B
    B --> N2
    N2 --> C
    C --> N3
    N3 -- Model2, Model4--> D
    D --> N4
    N4 --> E
    N3 -- Model1, Model3 --> E
    E --> N5
    N5 --> F
    F --> N6
    N6 -- Model3, Model4 --> G
    G --> N7
    N7 --> H
    N6 -- Model1, Model2 --> H
    H --> I
    FP1["CF_index, α"] -.-> B
    FP2["pref_dur, σ_dur"] -.-> D
    FP3["w, ρ(on/off ratio), τ_ON, τ_OFF"] -.-> G

    IN1@{ shape: lean-r}
    N1@{ shape: lean-r}
    N2@{ shape: lean-r}
    N3@{ shape: lean-r}
    N4@{ shape: lean-r}
    N5@{ shape: lean-r}
    N6@{ shape: lean-r}
    N7@{ shape: lean-r}
    I@{ shape: lean-r}
    L3@{ shape: lean-r}
     IN1:::io
     A:::perSeq
     N1:::io
     B:::perSeq
     N2:::io
     C:::perSeq
     N3:::io
     D:::perSeq
     N4:::io
     E:::perSeq
     N5:::io
     F:::runAssembly
     N6:::io
     G:::runAssembly
     N7:::io
     H:::runAssembly
     I:::output
     L1:::perSeq
     L2:::runAssembly
     L3:::io
     L4:::output
     FP:::param
     FP1:::param
     FP2:::param
     FP3:::param
    classDef perSeq fill:#eef2ff,stroke:#818cf8,color:#1e1b4b
    classDef runAssembly fill:#f0fdfa,stroke:#2dd4bf,color:#134e4a
    classDef output fill:#f5f3ff,stroke:#a78bfa,color:#4c1d95
    classDef io fill:#ecfeff,stroke:#22d3ee,color:#083344
    classDef param fill:#fefce8,stroke:#facc15,color:#713f12

```
