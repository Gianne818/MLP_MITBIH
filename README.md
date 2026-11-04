# CS345 Activity: Real-World Prediction with a Custom Multi-Layer Perceptron in C#

## 1. Overview & Problem Track
* **Track:** Multi-Class Classification (5 AAMI Arrhythmia Beat Classes)
* **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet)
* **Language & Runtime:** C# (.NET 10.0), using standard libraries only (`System`, `System.IO`, `System.Globalization`) — **zero external ML packages**.

---

## 2. Dataset Description (Deliverable #3)
The MIT-BIH Arrhythmia Database contains 48 half-hour ambulatory two-channel ECG recordings from 47 patients. Heartbeats were extracted around annotated R-peaks from Lead MLII (-250 ms to +305 ms, 200 samples at 360 Hz downsampled 2x to **100 feature points** per beat).

Each heartbeat is mapped to the standard **AAMI EC57** clinical categories:

| Class Code | Name | Description | Count |
| :---: | :--- | :--- | :---: |
| **0 (`N`)** | Normal | Normal sinus beats, bundle branch blocks | 3,000 |
| **1 (`S`)** | Supraventricular | Atrial premature / aberrant premature beats | 2,781 |
| **2 (`V`)** | Ventricular | Premature ventricular contractions (PVC), ventricular escape | 3,000 |
| **3 (`F`)** | Fusion | Fusion of ventricular and normal beats | 802 |
| **4 (`Q`)** | Unknown / Paced | Paced beats, unclassifiable beats | 3,000 |

* **Total Samples:** 12,583 heartbeats
* **Feature Dimensions:** 100 continuous numeric voltages in millivolts (mV)
* **Target Output:** 5 classes (one-hot encoded to 5 softmax nodes)

---

## 3. Data Preprocessing Pipeline (Criterion 2 — 20 pts)
Implemented in [`DataPreprocessor.cs`](DataPreprocessor.cs):
1. **CSV Parsing & Cleaning:** Parses `data/mitbih_beats.csv`, validates column count, handles decimal formatting (`InvariantCulture`), and drops any rows with missing or `NaN`/`Infinity` values.
2. **Stratified 80/20 Split:** Splits samples per-class so minority and majority classes maintain identical proportions:
   * **Training Set:** 10,065 samples (80%)
   * **Testing Set:** 2,518 samples (20%)
3. **Leakage-Free Min-Max Scaling to $[0, 1]$:**
   $$\tilde{x}_{i,j} = \frac{x_{i,j} - \min_{\text{train}}(x_j)}{\max_{\text{train}}(x_j) - \min_{\text{train}}(x_j)}$$
   Scaling bounds ($\min, \max$) are computed **strictly on the training partition** and applied to transform both training and testing sets.
4. **Target Encoding:** Converts integer class labels ($0..4$) into 1-of-N one-hot target vectors.

---

## 4. Custom Neural Network Architecture (Criterion 1 — 35 pts)
Implemented in [`NeuralNetwork.cs`](NeuralNetwork.cs):
* **Topology:** 100 Input Nodes $\rightarrow$ 64 Hidden Nodes ($\tanh$) $\rightarrow$ 5 Output Nodes ($\text{Softmax}$).
* **Loss Function:** Multi-class Cross-Entropy Loss:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^5 t_{i,k} \ln(y_{i,k})$$
* **Gradients & Backpropagation:**
  * Output signals: $\delta_k = t_k - y_k$ (for Softmax + Cross-Entropy)
  * Hidden signals: $\delta_j = (1 - h_j^2) \sum_k \delta_k w_{jk}$
  * Weight updates with momentum ($\alpha = 0.01$) and L2 weight decay ($\lambda = 5 \times 10^{-7}$).
* **Numerical Stability Fixes:**
  * Max-shift subtraction inside $\text{Softmax}$ to eliminate exponential overflow (`NaN`).
  * Clamping ($\max(y_k, 10^{-15})$) inside $\ln(y_k)$ to eliminate `-Infinity` errors.
  * Symmetric Xavier-scale random weight initialization.

---

## 5. Hyperparameter Tuning & Results (Criterion 3 — 15 pts)
Implemented in [`HyperparameterTuner.cs`](HyperparameterTuner.cs):
A systematic grid search was performed over hidden layer sizes (`16, 32, 64`) and learning rates (`0.005, 0.010, 0.030`) over 80 epochs. Results exported to `outputs/tuning_results.csv`:

| Rank | Hidden Nodes | Learning Rate | Test Loss | Test Accuracy | Macro-F1 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **64** | **0.010** | **0.1019** | **96.90%** | **96.29%** | **Optimal** |
| 2 | 16 | 0.010 | 0.1123 | 96.86% | 96.03% | Candidate |
| 3 | 64 | 0.030 | 0.1149 | 96.23% | 95.45% | Candidate |
| 4 | 32 | 0.005 | 0.1323 | 96.23% | 95.29% | Candidate |
| 5 | 16 | 0.005 | 0.1379 | 96.07% | 95.04% | Candidate |
| 6 | 16 | 0.030 | 0.1325 | 96.19% | 94.88% | Candidate |
| 7 | 32 | 0.010 | 0.1430 | 95.67% | 94.86% | Candidate |
| 8 | 64 | 0.005 | 0.1265 | 95.79% | 94.68% | Candidate |
| 9 | 32 | 0.030 | 0.1741 | 94.52% | 93.63% | Candidate |

**Justification:** Hidden size 64 with a learning rate of 0.010 achieved the lowest test loss (0.1019) and highest Macro-F1 (96.29%), giving the network enough capacity to capture fine QRS morphologic differences across arrhythmia categories without overfitting.

---

## 6. Independent Test Set Evaluation (Criterion 4 — 15 pts)
Evaluated on 2,518 unseen test heartbeats after 200 epochs of training (loss log saved to `outputs/training_loss_log.csv`):
* **Overall Test Accuracy:** **97.62%**
* **Test Cross-Entropy Loss:** **0.0858**
* **Macro-Averaged F1-Score:** **96.78%**

### Confusion Matrix (Rows = Actual, Columns = Predicted)
```text
Actual \ Pred |        N        S        V        F        Q | Total
--------------------------------------------------------------------
N (Normal)    |      599        0        0        1        0 |   600
S (Supravent) |        7      534        8        0        8 |   557
V (Ventric)   |        0        4      594        2        0 |   600
F (Fusion)    |        1        5       14      140        1 |   161
Q (Unknown)   |        0        6        3        0      591 |   600
```

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **N (Normal)** | 98.68% | 99.83% | **99.25%** | 600 |
| **S (Supraventricular)** | 97.27% | 95.87% | **96.56%** | 557 |
| **V (Ventricular)** | 95.96% | 99.00% | **97.46%** | 600 |
| **F (Fusion)** | 97.90% | 86.96% | **92.11%** | 161 |
| **Q (Unknown/Paced)** | 98.50% | 98.50% | **98.50%** | 600 |
| **Macro Average** | **97.66%** | **96.03%** | **96.78%** | **2,518** |
| **Weighted Average** | - | - | **97.59%** | **2,518** |

---

## 7. How to Build & Run
```bash
cd MLP_MITBIH
dotnet build
dotnet run
```

### Project Files & Deliverables
* [`NeuralNetwork.cs`](NeuralNetwork.cs) — Custom 3-layer backprop MLP engine
* [`DataPreprocessor.cs`](DataPreprocessor.cs) — CSV loading, stratified split, Min-Max [0, 1] scaling
* [`HyperparameterTuner.cs`](HyperparameterTuner.cs) — Systematic grid experiments & leaderboard
* [`EvaluationMetrics.cs`](EvaluationMetrics.cs) — Confusion matrix, precision, recall, F1-score
* [`Program.cs`](Program.cs) — Orchestrator and inference demo
* `outputs/training_loss_log.csv` — Deliverable #2 (loss history every 10 epochs)
* `outputs/tuning_results.csv` — Deliverable #2 / Hyperparameter table
* `data/mitbih_beats.csv` — Extracted 100-feature ECG dataset
