# MLP for Heartbeat Classification (MIT-BIH)

CS345 activity. A neural network written from scratch in C# (.NET 10) that classifies heartbeats into 5 types. It uses only the standard libraries, with no ML packages.

## Dataset

The data comes from the MIT-BIH Arrhythmia Database on PhysioNet. Each heartbeat is a 100-point slice of the ECG signal (Lead MLII) around the R-peak. The values are voltages in mV.

There are 12,583 heartbeats in 5 classes:

| Class | Meaning | Count |
| :--- | :--- | :---: |
| 0 (N) | Normal beats | 3,000 |
| 1 (S) | Supraventricular (premature atrial beats) | 2,781 |
| 2 (V) | Ventricular (PVCs) | 3,000 |
| 3 (F) | Fusion beats | 802 |
| 4 (Q) | Paced / unclassifiable | 3,000 |

## Preprocessing

Code is in `DataPreprocessor.cs`.

1. Load `data/mitbih_beats.csv` and skip any row with missing, NaN, or infinite values.
2. Split 80/20 into train and test, keeping the same class proportions in both (10,065 train, 2,518 test).
3. Scale every feature to the 0 to 1 range. The min and max come from the training set only, so no test information leaks in.
4. Turn each label into a one-hot vector of length 5.

## The Network

Code is in `NeuralNetwork.cs`.

- Structure: 100 inputs, 64 hidden nodes (tanh), 5 outputs (softmax)
- Loss: cross-entropy
- Training: backpropagation with momentum (0.01) and a small L2 weight decay (5e-7)
- Weights start as small random values scaled Xavier-style

To avoid numerical problems, softmax subtracts the max value before exponentiating, and the predicted probability is clamped to at least 1e-15 before taking the log.

## Hyperparameter Tuning

Code is in `HyperparameterTuner.cs`. I tried every combination of hidden size (16, 32, 64) and learning rate (0.005, 0.01, 0.03), training each for 80 epochs. Results are saved in `outputs/tuning_results.csv`, sorted here by Macro-F1.

| Hidden | Learning Rate | Test Loss | Accuracy | Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| **64** | **0.010** | **0.1019** | **96.90%** | **96.29%** |
| 16 | 0.010 | 0.1123 | 96.86% | 96.03% |
| 64 | 0.030 | 0.1149 | 96.23% | 95.45% |
| 32 | 0.005 | 0.1323 | 96.23% | 95.29% |
| 16 | 0.005 | 0.1379 | 96.07% | 95.04% |
| 16 | 0.030 | 0.1325 | 96.19% | 94.88% |
| 32 | 0.010 | 0.1430 | 95.67% | 94.86% |
| 64 | 0.005 | 0.1265 | 95.79% | 94.68% |
| 32 | 0.030 | 0.1741 | 94.52% | 93.63% |

I chose 64 hidden nodes with a learning rate of 0.01 because it had the lowest test loss and the best Macro-F1.

## Final Results

I trained the chosen setup for 200 epochs (loss history in `outputs/training_loss_log.csv`) and tested it on the 2,518 held-out heartbeats.

- Accuracy: 97.62%
- Test loss: 0.0858
- Macro-F1: 96.78%

Confusion matrix (rows are actual, columns are predicted):

```text
Actual \ Pred |   N     S     V     F     Q  | Total
-------------------------------------------------
N             |  599     0     0     1     0 |  600
S             |    7   534     8     0     8 |  557
V             |    0     4   594     2     0 |  600
F             |    1     5    14   140     1 |  161
Q             |    0     6     3     0   591 |  600
```

| Class | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
| N | 98.68% | 99.83% | 99.25% | 600 |
| S | 97.27% | 95.87% | 96.56% | 557 |
| V | 95.96% | 99.00% | 97.46% | 600 |
| F | 97.90% | 86.96% | 92.11% | 161 |
| Q | 98.50% | 98.50% | 98.50% | 600 |
| Macro avg | 97.66% | 96.03% | 96.78% | 2,518 |

Fusion beats (F) are the weakest class. Only 140 of 161 were caught, and 14 were mistaken for ventricular beats. That makes sense, since a fusion beat is partly a ventricular beat and F has the fewest training samples.

## Running It

```bash
cd MLP_MITBIH
dotnet build
dotnet run
```

## Files

- `NeuralNetwork.cs`: the MLP and backpropagation
- `DataPreprocessor.cs`: loading, splitting, scaling
- `HyperparameterTuner.cs`: grid search
- `EvaluationMetrics.cs`: confusion matrix, precision, recall, F1
- `Program.cs`: runs everything
- `outputs/training_loss_log.csv`: loss every 10 epochs
- `outputs/tuning_results.csv`: tuning results
- `data/mitbih_beats.csv`: the 100-feature dataset
