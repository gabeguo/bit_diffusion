# LaTeX for the rebuttal experiments

Drop-in LaTeX fragments describing the experiments queued by
`text_to_image/queue_rebuttal_experiments.sh` (see `text_to_image/REBUTTAL_EXPERIMENTS.md`).
Add `\input{rebuttal_experimental_description/macros.tex}` to the preamble (requires
`xcolor`). Any number not yet computed is a red `\tbd`. Fill it from the listed
result files; never estimate it.

| File | Reviewer | Where it goes | Results |
|---|---|---|---|
| `table2_foundation_models.tex` | 6E8U | Replaces Table 2 (§6.3): adds FlowTok-XL and CrossFlow rows | `external/{flowtok_xl,crossflow_256_clip}/results.json` |
| `external_baselines_setup.tex` | 6E8U | App. L (setup of those rows) | — |
| `cfg_roundtrip_editing.tex` | 4qvB Q2 | New appendix next to App. K/M/R; two figures | `roundtrip/compare_{image,text}.png`, `editing/cfg0.5/figure5_cfg0.5.pdf` |
| `shared_vs_separate.tex` | xYk4, 6E8U | New appendix / §6.3 | `bit_bidirectional/eval_{200000,400000}/*/eval/eval_metrics.json` |
| `shared_vs_separate_small_scale.tex` | xYk4 | Small-scale (DiTXA-B/2) version of the shared-vs-separate table | `bit_small_scale/<run>/eval_<step>/*/eval/eval_metrics.json` |
| `stochastic_endpoint_flow.tex` | 6E8U | §6.1 / Table 1 extension | `flow_comparison/abl_flow_matching_*/*/eval/eval_metrics.json` |
| `optional_n2d_side_information_roundtrip.tex` | 4qvB W1 | Optional, only if `roundtrip_n2d_cond` is run | `roundtrip/cfg*/n2d_cond_*/results.json` |

Cross-references to check against the paper source: `eq:cfg-forward` and
`eq:cfg-reverse` (Eqs. 10–11), `tab:foundation` (Table 2), and citation keys
`he2025flowtok`, `liu2025crossflow`, `albergo2024stochastic`.
