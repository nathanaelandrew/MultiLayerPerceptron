# Regression with a Custom 3-Layer MLP (C#)

Predicts a student's **FinalGrade** from **StudyHours** and **AttendanceRate** using only the
provided back-propagation classes. `BackNeural.cs`, `InputN.cs`, `HiddenN.cs` and `OuputN.cs`
are used **exactly as provided (unmodified)**; all new code is in `Program.cs`.

## How to run
Requires the [.NET 8 SDK](https://dotnet.microsoft.com/download).
```
dotnet run                          # uses dataset.csv copied next to the executable
dotnet run -- path/to/other.csv     # any CSV: numeric columns, LAST column = target
```
Everything is printed to the console: engine self-test, data summary, tuning table, per-epoch
loss log, metrics. `sample_console_output.txt` is one captured run.

## Project layout
| File | Role |
|---|---|
| `BackNeural.cs`, `InputN.cs`, `HiddenN.cs`, `OuputN.cs` | Provided engine, unmodified |
| `Program.cs` | `Program` (orchestration) plus `CsvLoader`, `MinMaxScaler`, `DataSplitter`, `RegressionModel`, `RegressionMetrics`, `HyperparameterTuner`, `EngineSelfTest` |
| `dataset.csv` | Data |

## Data description
`dataset.csv`: 8 hand-built student records, 2 features and 1 target, all on a 0-1 scale.

| Column | Meaning | Range | Correlation with FinalGrade |
|---|---|---|---|
| StudyHours | normalised weekly study time | 0.1 - 0.9 | 0.993 |
| AttendanceRate | fraction of classes attended | 0.2 - 0.9 | 0.966 |
| FinalGrade (target) | normalised final grade | 0.15 - 0.95 | - |

**Limitation:** 8 rows is far too few for a trustworthy test split. The code works for any size;
more rows (50+) is the biggest available improvement.

## Pre-processing
* Rows with missing or non-numeric values are dropped and counted.
* Features are min-max scaled to [0,1]; the target to [0.1,0.9] (the sigmoid output can only reach 0 or 1 asymptotically).
* Scalers are fitted on **training rows only** (no leakage); predictions are converted back to original units.
* 75/25 shuffled split (seed 42) -> 6 train / 2 test.

## Engine verification
The self-test checks the unmodified engine two ways: the forward pass against a hand formula (using the activity-sheet
inputs), and the weight update of one `learn()` step against a finite-difference gradient of the loss
(26 parameters, worst relative error ~1e-8). Since `NeuralNet` exposes no weight accessors, weights are read and
written through its own `saveWeights` / `loadWeights`.

## Hyperparameter experiments
12 configurations (hidden nodes {2,4,8,16} x epochs {500,2000,5000}), each trained from 3 random
initialisations and scored by leave-one-out CV on the training rows (test rows never used).
**Learning rates cannot be tuned**: they are `private const` in `BackNeural.cs` (output layer 0.2, input layer 0.15).

| Hidden | 500 epochs | 2000 epochs | 5000 epochs |
|---|---|---|---|
| 2 | 0.1176 | 0.0781 | 0.0747 |
| 4 | 0.1001 | 0.0700 | 0.0687 |
| 8 | 0.0867 | 0.0685 | 0.0663 |
| 16 | 0.0755 | 0.0674 | **0.0650** |

(CV RMSE, original units.) More hidden nodes helped steadily with diminishing returns (8 -> 16 only 2%); 500 epochs is
under-trained at fixed rates 0.2/0.15; error flattens after ~2000-5000 epochs. **Selected: 16 hidden nodes, 5000 epochs.**
The winner is on the grid edge, but the 8 -> 16 gain is small, so a larger grid is unlikely to change much.

## Results
| Set | RMSE | MAE | R² |
|---|---|---|---|
| Train (6 rows) | 0.018 | 0.015 | 0.988 |
| Test (2 rows) | 0.188 | 0.175 | -5.3 (meaningless with n=2) |
| Mean-predictor baseline on test | 0.464 | 0.458 | - |
| **LOOCV, all 8 rows** | **0.070** | **0.057** | **0.920** |

**Interpretation.** Training loss falls from ~0.08 to ~0.0008 (scaled MSE) and flattens: the network learned the trend.
The test error looks poor because the split placed both test rows (grades 0.30 and 0.15) *below* the lowest training
grade (0.45); a sigmoid network cannot extrapolate, so it predicts ~0.40 for both. The program detects and reports this.
LOOCV (every row predicted by a model that never saw it) gives R² = 0.92 and a typical error of ~0.06, the fairer
estimate here. Even on the unfavourable split the model is ~2.5x better than the baseline.

## Known limitations of using the engine unmodified
* Learning rates fixed (see above), so they are not part of the tuning.
* The engine's random initial weights are unseeded, so exact numbers vary slightly between runs (shuffling is seeded).
* `countgood()` is not used as a stopping rule (its test, EC - DER != 0, is practically never false); training runs for a fixed number of epochs.