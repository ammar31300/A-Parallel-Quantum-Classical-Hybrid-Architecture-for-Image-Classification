# A Parallel Quantum-Classical Hybrid Architecture for Image Classification

**A parallel hybrid quantum-classical framework for image classification using convolutional feature extraction and multiple parameterized quantum neural network branches.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20ML-6c5ce7)](https://pennylane.ai/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Quantum](https://img.shields.io/badge/Quantum-Hybrid-purple)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This repository presents a **Parallel Quantum-Classical Hybrid Neural Network (PQCHNN)** for image classification.

The central idea is to combine the strengths of classical convolutional neural networks with multiple parallel quantum neural network (QNN) branches.

Instead of feeding CNN features into a single quantum circuit, the proposed architecture creates several **parallel quantum heads**. Each head processes a representation of the learned classical features using a specific quantum encoding or feature-organization strategy.

The resulting quantum representations are measured, concatenated, and passed to a classical classification head.

The general pipeline is:

```text
                         Input Image
                              │
                              ▼
                  ┌──────────────────────┐
                  │   Classical CNN      │
                  │   Feature Extraction │
                  └──────────┬───────────┘
                             │
                             ▼
                    Classical Feature Map
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
      ┌────────────┐  ┌────────────┐  ┌────────────┐
      │ Quantum    │  │ Quantum    │  │ Quantum    │
      │ Head 1     │  │ Head 2     │  │ Head 3     │
      │ Amplitude  │  │ Dense      │  │ ZZ Feature │
      │ Encoding   │  │ Angle      │  │ Encoding   │
      └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
            │               │                │
            ▼               ▼                ▼
       Quantum           Quantum          Quantum
       Features          Features         Features
            │               │                │
            └───────────────┼────────────────┘
                            ▼
                    Feature Concatenation
                            │
                            ▼
                    Classical Classifier
                            │
                            ▼
                         Prediction
```

The architecture is evaluated across image datasets with substantially different visual characteristics, including **MNIST**, **CIFAR-10**, and **SeekThermal**.

---

# 🎯 Motivation

Classical CNNs are highly effective at extracting spatial representations from images. Quantum machine learning provides an alternative mechanism for transforming learned features using quantum states, parameterized rotations, entanglement, and measurements.

This project investigates the following research question:

> **Can multiple parallel quantum feature transformations provide complementary representations of classical CNN features for image classification?**

The framework therefore combines:

* **CNNs** for spatial feature extraction
* **Quantum circuits** for trainable feature transformations
* **Parallel quantum heads** for multiple processing pathways
* **Quantum measurements** for converting quantum states into classical features
* **Classical classifiers** for final prediction

The objective is not to assume or claim quantum advantage, but to experimentally investigate whether parallel quantum processing can provide useful representations within a hybrid image-classification architecture.

---

# 🧠 Main Contributions

The project combines several components into a single experimental framework.

### 1. Classical Feature Extraction

A convolutional neural network extracts spatial representations from the input image.

### 2. Parallel Quantum Processing

Instead of using a single quantum circuit, multiple quantum heads process the learned representation:

```text
                 CNN Feature Representation
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
           QNN-1         QNN-2         QNN-3
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                  Quantum Feature Fusion
```

### 3. Multiple Quantum Encodings

The implementation investigates:

* Amplitude Encoding
* Dense Angle Encoding
* ZZ Feature Encoding

### 4. Spatial Feature Permutations

The amplitude-based architecture can expose different spatial organizations of the same CNN feature map to different quantum heads.

### 5. Structured Entanglement

The quantum circuits support structured connectivity patterns including:

* Ring
* Pairwise
* Ring + Skip

### 6. Variational Quantum Circuits

Each quantum branch contains trainable quantum parameters and data-dependent encoding operations.

### 7. Classical Baseline

A fully classical model is trained as a reference point for evaluating the hybrid architectures.

### 8. Multi-Dataset Evaluation

The framework is evaluated on:

* MNIST
* CIFAR-10
* SeekThermal

### 9. Reproducible Evaluation

The experiments include:

* Accuracy
* Macro-F1
* Weighted-F1
* Classification reports
* Confusion matrices
* Training/validation curves
* Checkpointing
* Early stopping
* Fixed random seeds

---

# 🏗️ Architecture

## High-Level Architecture

The complete hybrid architecture is:

```text
                         Input Image
                              │
                              ▼
                  ┌──────────────────────┐
                  │    Classical CNN     │
                  │      Backbone        │
                  └──────────┬───────────┘
                             │
                             ▼
                    Feature Representation
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        Quantum Head 1  Quantum Head 2  Quantum Head 3
              │              │              │
              ▼              ▼              ▼
          Encoding       Encoding       Encoding
              │              │              │
              ▼              ▼              ▼
         Variational     Variational     Variational
           Circuit         Circuit         Circuit
              │              │              │
              ▼              ▼              ▼
         Measurement     Measurement     Measurement
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    Feature Concatenation
                             │
                             ▼
                    Classical Classifier
                             │
                             ▼
                          Prediction
```

---

# 🔬 Classical CNN Backbone

The classical backbone is responsible for extracting spatial representations before quantum processing.

A representative convolutional pipeline is:

```text
Input
  │
  ├── Conv2D
  ├── BatchNorm
  ├── ReLU
  ├── MaxPool
  │
  ├── Conv2D
  ├── BatchNorm
  ├── ReLU
  ├── MaxPool
  │
  ├── Conv2D
  ├── BatchNorm
  ├── ReLU
  ├── MaxPool
  │
  └── Conv2D
      ├── BatchNorm
      └── ReLU
```

The SeekThermal implementation uses approximately the following channel progression:

```text
Input
  ↓
16 channels
  ↓
32 channels
  ↓
64 channels
  ↓
64 channels
```

Depending on the experiment, the backbone can provide both:

* a compact feature vector
* a spatial feature representation

The spatial representation is particularly relevant to the amplitude-encoding branch.

---

# ⚛️ Parallel Quantum Neural Network

The quantum component is organized around multiple quantum processing branches.

Conceptually:

```text
Classical Feature Representation
              │
              ▼
       Head-specific Preparation
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
      QNode  QNode  QNode
        1      2      3
       │      │      │
       └──────┼──────┘
              ▼
      Quantum Feature Vector
```

Each quantum head can have:

* an independent quantum node
* trainable variational parameters
* a head-specific feature representation
* a specific encoding strategy
* a structured entanglement pattern

For the main six-qubit configuration, each head produces expectation-value measurements from six qubits.

---

# 🔀 Parallel Head Configuration

The default SeekThermal configuration uses:

| Parameter            |    Value |
| -------------------- | -------: |
| Parallel heads       |        3 |
| Qubits per head      |        6 |
| Quantum layers       |        3 |
| Measured observables | \(X, Z\) |

Each qubit contributes two expectation values:

$$
\langle X_i\rangle,\qquad
\langle Z_i\rangle
$$

Therefore, each quantum head produces:

$$
6\times2=12
$$

features.

With three parallel heads:

$$
3\times6\times2=36
$$

quantum features are produced before the final classifier.

---

# ⚛️ Quantum Encoding Strategies

The repository investigates three primary quantum processing variants:

| Model                      | Quantum Encoding     | Main Representation                          |
| -------------------------- | -------------------- | -------------------------------------------- |
| `HybridAmplitudeParallel`  | Amplitude Encoding   | CNN spatial feature map                      |
| `HybridDenseAngleParallel` | Dense Angle Encoding | Classical feature vector                     |
| `HybridZZFeatureParallel`  | ZZ Feature Encoding  | Feature-dependent rotations and interactions |

---

# 1. `HybridAmplitudeParallel`

## Amplitude Encoding

The amplitude-based architecture converts a classical feature vector into the amplitudes of a quantum state.

For \(n\) qubits, the Hilbert-space dimension is:

$$
D=2^n
$$

For six qubits:

$$
2^6=64
$$

Therefore, the quantum representation uses 64 amplitudes.

The processing pipeline is:

```text
CNN Feature Map
      │
      ▼
Spatial Permutation
      │
      ▼
Adaptive Average Pooling
      │
      ▼
Flatten
      │
      ▼
Linear Projection
      │
      ▼
64-dimensional Vector
      │
      ▼
Amplitude Embedding
```

The quantum state is represented as:

$$
|\psi(x)\rangle
=
\sum_{i=0}^{D-1}x_i|i\rangle
$$

subject to:

$$
\sum_i |x_i|^2=1.
$$

The implementation performs normalization and zero-padding where necessary.

A representative PennyLane operation is:

```python
qml.AmplitudeEmbedding(
    features=amp_batch,
    wires=range(n_qubits),
    normalize=True,
    pad_with=0.0,
)
```

---

# 🔢 Spatial Feature Permutations

A key component of the amplitude architecture is the use of deterministic spatial permutations.

Different quantum heads can process different arrangements of the same CNN feature map.

Supported permutation families include:

* Identity
* Serpentine
* Checkerboard
* Row shift
* Column shift
* Reverse
* Affine permutation
* Diagonal wrapping

Conceptually:

```text
                  CNN Feature Map
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Permutation 1  Permutation 2  Permutation 3
          │              │              │
          ▼              ▼              ▼
        QNN-1          QNN-2          QNN-3
```

The permutations do not change the feature values themselves. They modify the ordering in which the spatial information is presented to the quantum encoder.

This provides representation diversity between parallel quantum heads.

---

# 2. `HybridDenseAngleParallel`

## Dense Angle Encoding

The dense-angle architecture maps classical features to quantum rotation angles.

A representative transformation is:

$$
\theta_i=\pi\tanh(x_i).
$$

The `tanh` transformation bounds the input before conversion to rotation angles.

The processing pipeline is:

```text
CNN Features
      │
      ▼
Feature Projection / Reshaping
      │
      ▼
Angle Transformation
      │
      ▼
RX / RY Encoding
      │
      ▼
Data-dependent Interaction
      │
      ▼
Variational Rotations
      │
      ▼
Entanglement
      │
      ▼
Measurement
```

Typical data-encoding operations include:

$$
R_X(\theta_i)
\qquad
R_Y(\theta_i).
$$

The quantum circuit then applies trainable variational operations and structured entanglement.

---

# 3. `HybridZZFeatureParallel`

## ZZ Feature Encoding

The ZZ-based architecture introduces feature-dependent two-qubit interactions.

For qubits \(i\) and \(j\):

$$
U_{ZZ}(\phi_{ij})
=
e^{-i\phi_{ij}Z_iZ_j}.
$$

A common circuit decomposition is:

```text
q_i ──●────RZ(φ)────●──
      │              │
q_j ──X──────────────X──
```

The implementation uses a configurable interaction scale:

```python
zz_scale = 0.25
```

The branch combines:

* single-qubit data encoding
* feature-dependent ZZ interactions
* trainable rotations
* entanglement
* quantum measurements

This allows feature information to influence both individual qubit states and correlations between qubits.

---

# 🔗 Variational Quantum Circuit

The quantum branches use trainable variational parameters in addition to data-dependent encoding operations.

A general single-qubit rotation is:

$$
Rot(\alpha,\beta,\gamma)
=
R_Z(\gamma)
R_Y(\beta)
R_Z(\alpha).
$$

A representative quantum layer is:

$$
U_l(\mathbf{x},\theta_l)
=
U_{\mathrm{ent}}
U_{\mathrm{var}}(\theta_l)
U_{\mathrm{enc}}(\mathbf{x}).
$$

For \(L\) quantum layers:

$$
U(\mathbf{x},\Theta)
=
U_L\cdots U_2U_1.
$$

A typical layer therefore contains:

```text
Data Encoding
      │
      ▼
Parameterized Rotations
      │
      ▼
Entanglement
      │
      ▼
Parameterized Rotations
```

The SeekThermal configuration uses:

$$
n_{\mathrm{qubits}}=6
$$

and

$$
n_{\mathrm{layers}}=3.
$$

---

# 🔗 Entanglement Patterns

The implementation supports several structured entanglement patterns.

## Ring Entanglement

A cyclic nearest-neighbor topology:

$$
q_0-q_1-q_2-\cdots-q_{n-1}-q_0
$$

provides local interactions while closing the topology into a ring.

---

## Pairwise Entanglement

Qubits can be grouped into structured pairs:

```text
q0 ── q1

q2 ── q3

q4 ── q5
```

The interaction pattern can be shifted between layers.

---

## Ring + Skip

The circuit can combine nearest-neighbor interactions with longer-range skip connections.

Conceptually:

```text
q0 ─── q1 ─── q2 ─── q3
│                   │
└───────────────────┘
```

This increases connectivity while maintaining a structured circuit.

---

# 🔄 Layer-Dependent Connectivity

The entanglement pattern can be modified between quantum layers.

A layer-dependent shift can be represented conceptually as:

```python
current_shift = layer % n_qubits
```

This allows different layers to expose different interaction patterns without requiring an entirely different circuit architecture for every layer.

---

# 📏 Quantum Measurements

The final quantum representation is extracted through expectation-value measurements.

For every qubit, the implementation can measure:

$$
\langle X_i\rangle
$$

and

$$
\langle Z_i\rangle.
$$

For six qubits:

$$
6\times2=12
$$

measurements are produced per quantum head.

For three heads:

$$
3\times12=36.
$$

Therefore:

```text
3 heads
×
6 qubits
×
2 observables
=
36 quantum features
```

The resulting classical feature vector is:

$$
Q\in\mathbb{R}^{36}.
$$

---

# 🧮 Final Classification Head

The fused quantum representation is passed to a classical classifier.

For the default configuration:

```text
Quantum Features
      │
      ▼
36-dimensional Vector
      │
      ▼
Linear Classifier
      │
      ▼
Class Logits
```

Mathematically:

$$
\hat{y}=WQ+b.
$$

For the three-class SeekThermal experiment:

$$
\mathbb{R}^{36}
\rightarrow
\mathbb{R}^{3}.
$$

---

# 🆚 Classical Baseline

A fully classical baseline is included to provide a reference point.

The baseline follows the same general CNN feature-extraction philosophy but removes the quantum component.

Conceptually:

```text
Image
  │
  ▼
CNN
  │
  ▼
Classical Feature Vector
  │
  ▼
MLP / Linear Classifier
  │
  ▼
Prediction
```

This comparison is important because hybrid models should be evaluated against a comparable classical architecture rather than only by their absolute performance.

---

# 📊 Experimental Models

The main experimental model families are:

| Model                      | CNN | Quantum Module       | Parallel Heads |
| -------------------------- | :-: | -------------------- | -------------: |
| `HybridAmplitudeParallel`  |  ✓  | Amplitude Encoding   |              3 |
| `HybridDenseAngleParallel` |  ✓  | Dense Angle Encoding |              3 |
| `HybridZZFeatureParallel`  |  ✓  | ZZ Feature Encoding  |              3 |
| `FullClassicalBaseline`    |  ✓  | None                 |              0 |

---

# 🗂️ Datasets

The framework is evaluated on three image-classification settings.

---

## 1. MNIST

MNIST provides a relatively simple grayscale image-classification benchmark.

| Property      | Value              |
| ------------- | ------------------ |
| Domain        | Handwritten digits |
| Image type    | Grayscale          |
| Original size | 28 × 28            |
| Model input   | 32 × 32            |
| Channels      | 1                  |
| Classes       | 10                 |

The images are resized to \(32\times32\) and normalized using:

$$
\mu=0.5,\qquad\sigma=0.5.
$$

The demonstrated configuration uses a selected subset of the available training data:

| Split      | Samples |
| ---------- | ------: |
| Training   |  10,800 |
| Validation |   1,200 |
| Test       |  10,000 |

---

## 2. CIFAR-10

CIFAR-10 provides a natural RGB image-classification benchmark.

| Property     | Value          |
| ------------ | -------------- |
| Domain       | Natural images |
| Image type   | RGB            |
| Image size   | 32 × 32        |
| Channels     | 3              |
| Classes      | 10             |
| Training set | 50,000         |
| Test set     | 10,000         |

Classes:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

Training augmentation includes:

* Random horizontal flip
* Random crop
* Padding
* Tensor conversion
* Dataset normalization

The normalization parameters are:

$$
\mu=(0.4914,0.4822,0.4465)
$$

and

$$
\sigma=(0.2023,0.1994,0.2010).
$$

### Reported CIFAR-10 Experiment

The notebook reports the following results for the specified experimental configuration:

| Model                     | Test Accuracy | Macro-F1 | Weighted-F1 | Test Loss | Best Validation Accuracy |
| ------------------------- | ------------: | -------: | ----------: | --------: | -----------------------: |
| `HybridAmplitudeParallel` |        0.7236 |   0.7211 |      0.7211 |    0.7965 |                   0.6942 |
| `FullClassicalBaseline`   |        0.7327 |   0.7314 |      0.7314 |    0.7605 |                   0.7004 |

Training duration:

$$
15\text{ epochs}.
$$

These values describe the reported experimental run and should not be interpreted as universal performance claims.

---

## 3. SeekThermal

SeekThermal extends the evaluation to thermal object imagery.

The experiment considers three object classes:

```text
Car
Cat
Man
```

The dataset follows a structure similar to:

```text
SeekThermal/
├── Train/
│   ├── Car/
│   ├── Cat/
│   └── Man/
│
└── Test/
    ├── car/
    ├── cat/
    └── man/
```

The reported dataset counts are:

| Class     | Training Images | Test Images |
| --------- | --------------: | ----------: |
| Car       |           1,168 |         356 |
| Cat       |           1,782 |         356 |
| Man       |           1,782 |         356 |
| **Total** |       **4,732** |   **1,068** |

Images are resized to:

$$
128\times96
$$

and processed using three input channels in the implementation.

Normalization:

$$
\mu=(0.5,0.5,0.5)
$$

$$
\sigma=(0.5,0.5,0.5).
$$

---

# 🧪 SeekThermal Train / Validation / Test Split

The training set is divided using a stratified split.

The validation ratio is:

$$
r_{\mathrm{val}}=0.15.
$$

The split uses:

```text
Random seed = 42
```

Stratification preserves the class distribution as closely as possible between training and validation subsets.

The independent test set remains isolated from model optimization.

```text
Training Directory
        │
        ├──────── 85% ────────► Training
        │
        └──────── 15% ────────► Validation

Test Directory
        │
        └──────────────────────► Final Evaluation
```

---

# ⚠️ SeekThermal Label-Mapping Issue

An important preprocessing issue was identified in the SeekThermal experiment.

The original `ImageFolder` mapping was:

```text
Car → 0
Cat → 1
Man → 2
```

Inspection of the directory contents indicated that the semantic contents of the `Car` and `Cat` folders were exchanged.

The notebook therefore applies an explicit label correction:

$$
0\rightarrow1
$$

$$
1\rightarrow0
$$

$$
2\rightarrow2.
$$

Equivalent implementation:

```python
remap = {
    0: 1,
    1: 0,
    2: 2,
}
```

This correction is important for reproducibility and should remain explicitly documented in the data pipeline.

---

# ⚙️ Experimental Configuration

A representative SeekThermal configuration is:

| Parameter         |       Value |
| ----------------- | ----------: |
| Random Seed       |          42 |
| Image Height      |         128 |
| Image Width       |          96 |
| Number of Classes |           3 |
| Batch Size        |          64 |
| Epochs            |          10 |
| Learning Rate     | \(10^{-3}\) |
| Weight Decay      | \(10^{-4}\) |
| Number of Qubits  |           6 |
| Quantum Layers    |           3 |
| Parallel Heads    |           3 |
| ZZ Scale          |        0.25 |
| Validation Ratio  |        0.15 |

The exact configuration may differ between datasets and experiments.

---

# ⚛️ Quantum Simulation

The experiments use PennyLane's `default.qubit` simulator.

Conceptually:

```python
qml.device(
    "default.qubit",
    wires=n_qubits,
    shots=None
)
```

The experiments therefore use a **classical quantum simulator**, rather than a physical quantum processor.

This distinction is important when interpreting computational cost and claims about quantum advantage.

---

# 🏋️ Training Procedure

The hybrid models are trained end-to-end.

The training pipeline is:

```text
Input Batch
    │
    ▼
CNN Forward Pass
    │
    ▼
Classical Feature Representation
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
 Quantum Head 1  Quantum Head 2  Quantum Head 3
    │              │              │
    └──────────────┼──────────────┘
                   ▼
           Quantum Feature Fusion
                   │
                   ▼
           Classification Head
                   │
                   ▼
                  Loss
                   │
                   ▼
            Backpropagation
                   │
                   ▼
        Classical + Quantum Updates
```

The training framework supports:

* Adam / AdamW optimization
* Learning-rate scheduling
* Gradient clipping
* Early stopping
* Best-model checkpointing
* Validation monitoring
* Reproducible random seeds

---

# ⏹️ Checkpointing & Early Stopping

The best model is selected according to validation performance.

A typical checkpoint naming convention is:

```text
{name}_best.pth
```

This ensures that the model restored for final testing corresponds to the best validation state rather than necessarily the final training epoch.

---

# 📏 Evaluation Metrics

The project evaluates models using several complementary metrics.

## Accuracy

$$
\mathrm{Accuracy}
=
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
$$

## Macro-F1

$$
F1_{\mathrm{macro}}
=
\frac{1}{K}
\sum_{k=1}^{K}F1_k.
$$

Macro-F1 gives equal importance to every class.

## Weighted-F1

$$
F1_{\mathrm{weighted}}
=
\sum_{k=1}^{K}w_kF1_k.
$$

Weighted-F1 accounts for the number of samples belonging to each class.

## Additional Evaluation

The experiments also generate:

* Precision
* Recall
* Per-class F1
* Classification reports
* Confusion matrices
* Training curves
* Validation curves

---

# 📊 Results

The repository records experiment results for the different model variants.

A typical comparison has the following form:

| Model                      | Accuracy | Macro-F1 | Weighted-F1 | Loss |
| -------------------------- | -------: | -------: | ----------: | ---: |
| `HybridAmplitudeParallel`  |        — |        — |           — |    — |
| `HybridDenseAngleParallel` |        — |        — |           — |    — |
| `HybridZZFeatureParallel`  |        — |        — |           — |    — |
| `FullClassicalBaseline`    |        — |        — |           — |    — |

The exact values should be generated from the corresponding experiment outputs rather than manually copied into the README.

For the reported CIFAR-10 experiment, the available recorded values are provided in the dataset section above.

---

# 📈 Training & Visualization

The evaluation pipeline can generate:

* Training loss curves
* Validation loss curves
* Training accuracy curves
* Validation accuracy curves
* Macro-F1 curves
* Weighted-F1 curves
* Confusion matrices
* Model comparison plots

Example output structure:

```text
results/
├── metrics/
├── figures/
└── confusion_matrices/
```

---

# 🔁 Reproducibility

The primary experiments use:

```python
SEED = 42
```

The random seed is applied to the relevant Python, NumPy, and PyTorch components.

For reproducible experiments, the following should be recorded:

* Dataset version
* Dataset split
* Random seed
* Image preprocessing
* Data augmentation
* CNN architecture
* Number of qubits
* Number of quantum layers
* Number of quantum heads
* Encoding strategy
* Entanglement topology
* Measurement strategy
* Optimizer
* Learning rate
* Batch size
* Number of epochs
* Early-stopping configuration
* Simulator/backend
* Hardware/software environment

Exact reproducibility can still depend on the underlying hardware, software versions, and simulator implementation.

---

# 🧪 Research Questions

The architecture is designed to investigate several research questions.

### RQ1 — Parallel Quantum Processing

How does parallel quantum processing behave compared with a single quantum branch?

### RQ2 — Encoding Strategy

How do amplitude, dense-angle, and ZZ-based encodings affect the learned representation?

### RQ3 — Spatial Organization

Does changing the ordering of classical spatial features affect the resulting quantum representation?

### RQ4 — Dataset Dependence

Does the behavior of the hybrid architecture change across grayscale, natural RGB, and thermal imagery?

### RQ5 — Classical Comparison

How does the hybrid architecture compare with a comparable classical feature-extraction pipeline?

### RQ6 — Computational Cost

How does increasing quantum circuit complexity affect computational cost and model behavior?

---

# ⚖️ Hybrid vs. Classical Learning

The project is designed as an experimental comparison rather than a claim of universal quantum superiority.

The classical baseline addresses:

> How well can the task be solved using the classical representation alone?

The hybrid models address:

> What changes when learned CNN representations are transformed using parameterized quantum circuits?

A meaningful comparison should therefore consider more than accuracy:

```text
Accuracy
Macro-F1
Weighted-F1
Loss
Parameter Count
Training Time
Inference Cost
Convergence
```

---

# 🚧 Limitations

## 1. Quantum Simulation

The experiments use quantum simulation on classical hardware.

As the number of qubits, layers, heads, and circuit evaluations increases, simulation cost can grow substantially.

## 2. No Demonstrated Quantum Speedup

The use of a quantum simulator does not establish quantum computational advantage.

The purpose of the project is to investigate the architecture and its empirical behavior.

## 3. Limited Quantum Width

The main SeekThermal configuration uses:

```text
6 qubits
```

This keeps the circuit computationally manageable for simulation.

## 4. Dataset-Specific Configuration

Different datasets may require different:

* CNN backbones
* input dimensions
* normalization
* feature projections
* quantum encodings
* training configurations

Therefore, numerical comparisons across datasets should be interpreted within their respective experimental settings.

## 5. Need for Repeated Experiments

Single-run results can be sensitive to initialization and training stochasticity.

Future comparisons should ideally report results over multiple random seeds together with mean and standard deviation.

---

# 🔮 Future Work

Potential extensions include:

### Real Quantum Hardware

Evaluate the circuits on actual quantum processors and compare simulator and hardware behavior.

### Larger Quantum Circuits

Investigate the effect of increasing:

* qubit count
* circuit depth
* number of quantum heads

### Noise-Aware Training

Study the effect of:

* gate noise
* depolarizing noise
* bit-flip noise
* measurement noise

### Improved Feature Fusion

Instead of direct concatenation, investigate:

* Learnable fusion
* Attention-based fusion
* Gating mechanisms
* Weighted quantum heads
* Classical-quantum feature fusion

### Automated Circuit Search

Search over:

* Encoding strategies
* Entanglement topologies
* Circuit depth
* Measurement observables
* Number of qubits
* Number of parallel heads

### Stronger Classical Baselines

Future experiments could compare against architectures such as:

* ResNet
* MobileNet
* EfficientNet
* Vision Transformers
* Lightweight CNNs

---

# 💻 Installation

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install the core dependencies:

```bash
pip install torch torchvision
pip install pennylane
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pillow
pip install jupyter
```

Or, if available:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Experiments

Launch Jupyter:

```bash
jupyter notebook
```

Then open the corresponding notebook.

Example:

```text
notebooks/
├── MNIST/
├── CIFAR10/
└── SeekThermal/
```

Update the dataset path where required:

```python
dataset_root = "PATH_TO_DATASET"
```

Then execute the notebook cells sequentially.

---

# 🔬 Experimental Workflow

The complete workflow is:

```text
1. Set random seed
        │
        ▼
2. Load dataset
        │
        ▼
3. Apply preprocessing
        │
        ▼
4. Create train/validation split
        │
        ▼
5. Build DataLoaders
        │
        ▼
6. Build CNN backbone
        │
        ▼
7. Build quantum branches
        │
        ▼
8. Build classical baseline
        │
        ▼
9. Train models
        │
        ▼
10. Monitor validation performance
        │
        ▼
11. Apply early stopping
        │
        ▼
12. Restore best checkpoint
        │
        ▼
13. Evaluate on test set
        │
        ▼
14. Generate reports
        │
        ▼
15. Generate figures
        │
        ▼
16. Save experiment results
```

---

# 📁 Recommended Project Structure

A clean repository structure can follow:

```text
A-Parallel-Quantum-Classical-Hybrid-Architecture-for-Image-Classification/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── notebooks/
│   ├── MNIST/
│   ├── CIFAR10/
│   └── SeekThermal/
│
├── src/
│   ├── models/
│   │   ├── classical.py
│   │   ├── amplitude.py
│   │   ├── angle.py
│   │   └── zz_feature.py
│   │
│   ├── quantum/
│   │   ├── encodings.py
│   │   ├── entanglement.py
│   │   └── measurements.py
│   │
│   ├── data/
│   │   ├── mnist.py
│   │   ├── cifar10.py
│   │   └── seekthermal.py
│   │
│   └── training/
│       ├── train.py
│       └── evaluate.py
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── confusion_matrices/
│
├── checkpoints/
│
└── configs/
    ├── mnist.yaml
    ├── cifar10.yaml
    └── seekthermal.yaml
```

---

# 📚 Scientific Context

Hybrid quantum-classical neural networks combine conventional deep-learning components with parameterized quantum circuits.

The broader research area includes:

* Variational Quantum Circuits
* Quantum Neural Networks
* Quantum Convolutional Neural Networks
* Quantum-Classical Hybrid Learning
* Quantum Feature Maps
* Parameterized Quantum Circuits

The architecture presented in this repository focuses specifically on **parallel quantum feature processing**, where multiple quantum branches operate on related classical representations before their outputs are fused.

This design provides an experimental framework for studying whether increasing the **width of quantum processing** can offer useful representational diversity without relying solely on deeper quantum circuits.

---

# 📖 References

The implementation builds upon concepts from classical deep learning and quantum machine learning, including:

* PyTorch — Deep learning framework
* PennyLane — Quantum machine learning framework
* Parameterized quantum circuits
* Variational quantum algorithms
* Quantum feature maps
* Quantum-classical hybrid neural networks

Additional project-specific references can be added here as the associated research manuscript is finalized.

---

# 📝 Citation

If this repository is used in academic work, please cite the corresponding project or publication.

A repository-level BibTeX entry can be added as follows:

```bibtex
@software{parallel_quantum_classical_image_classification,
  title  = {A Parallel Quantum-Classical Hybrid Architecture for Image Classification},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/your-username/your-repository}
}
```

---

# ⭐ Key Takeaway

The core idea of the project can be summarized as:

```text
Classical CNN
      │
      ▼
Learned Spatial Representation
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
   Quantum         Quantum        Quantum
    Head 1          Head 2         Head 3
      │              │              │
      └──────────────┼──────────────┘
                     ▼
             Quantum Measurements
                     │
                     ▼
             Feature Concatenation
                     │
                     ▼
              Classical Classifier
                     │
                     ▼
                  Prediction
```

The project therefore provides a framework for studying how **parallel parameterized quantum circuits can be integrated with convolutional feature extraction for image classification**.

The combination of multiple encoding strategies, structured entanglement, spatial feature organization, and classical baselines makes the framework suitable for controlled experimentation across heterogeneous image domains.

---

# 📌 Reproducibility Note

Numerical results should always be interpreted together with the exact:

* Dataset version
* Dataset split
* Preprocessing pipeline
* Random seed
* CNN configuration
* Number of qubits
* Quantum circuit depth
* Number of quantum heads
* Encoding strategy
* Entanglement pattern
* Measurement strategy
* Optimizer
* Learning rate
* Training duration
* Simulator/backend

Changing these settings can affect the resulting performance.

---

# 📄 License

If this repository is released under the MIT License, the corresponding `LICENSE` file should be included at the repository root.

```text
MIT License
```

Otherwise, replace this section with the actual license associated with the project.
