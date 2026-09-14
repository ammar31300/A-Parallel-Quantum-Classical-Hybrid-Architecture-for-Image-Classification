# A Parallel Quantum-Classical Hybrid Architecture for Image Classification

<p align="center">

**A Parallel Quantum-Classical Hybrid Neural Architecture for Image Classification**

</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20Machine%20Learning-8c52ff.svg)](https://pennylane.ai/)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

</p>

---

## Overview

This repository implements a **parallel quantum-classical hybrid architecture for image classification**.

The proposed approach combines a lightweight **classical convolutional feature extractor** with multiple **parallel variational quantum neural network (QNN) heads**. Instead of sending the complete classical representation through a single quantum circuit, the architecture distributes the representation across multiple quantum processing paths and concatenates their measurement outputs before the final classification layer.

The implementation is designed to investigate how different classical-to-quantum encoding strategies and entanglement topologies affect image-classification performance and quantum-circuit properties.

The main architecture consists of:

```text
Input Image
     │
     ▼
┌───────────────────────┐
│ Classical CNN         │
│ Feature Extraction    │
└──────────┬────────────┘
           │
           ▼
    Classical Features
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
 Quantum Head 1  Quantum Head 2  ... Quantum Head P
     │           │                    │
     └───────────┴────────────────────┘
                 │
                 ▼
       Quantum Measurement
          <X> and <Z>
                 │
                 ▼
       Concatenated Quantum
          Feature Vector
                 │
                 ▼
        Classical Classifier
                 │
                 ▼
        Class Predictions
```

The notebook also provides experiments for:

* Amplitude embedding
* ZZ-based feature encoding
* Dense-angle encoding
* Multiple entanglement topologies
* Dynamic entanglement routing / imprinting
* Quantum expressibility
* Meyer–Wallach entanglement
* Classical-vs-hybrid benchmarking
* Training and validation convergence
* Test-set evaluation
* Confusion matrices
* Publication-ready circuit and architecture figures

---

# Motivation

Quantum machine learning models often face a fundamental bottleneck: current quantum devices have a limited number of qubits, while image data can contain thousands of classical features.

A direct mapping from an image to a quantum circuit is therefore impractical for many realistic image-classification problems.

This project addresses the problem through a **hybrid architecture**:

1. A classical CNN extracts compact visual representations.
2. The resulting representation is transformed into quantum-compatible features.
3. Multiple quantum circuits process the representation in parallel.
4. Each quantum head uses a potentially different entanglement topology.
5. Quantum measurements are concatenated.
6. A classical classifier converts the resulting quantum feature vector into class logits.

The parallel design provides multiple quantum processing paths while keeping the number of qubits per circuit relatively small.

---

# Key Ideas

## 1. Classical Feature Extraction

The classical backbone is a lightweight CNN implemented in PyTorch.

For a `32 × 32` input image, the encoder contains two convolutional blocks:

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
  └── MaxPool
```

The implementation uses:

* First convolution: `C → 8` channels
* Second convolution: `8 → 16` channels
* `3 × 3` kernels
* Batch normalization
* ReLU activations
* `2 × 2` max pooling
* Adaptive average pooling

For the standard image size used in the experiments, the convolutional feature map becomes:

```text
(B, 16, 8, 8)
```

The classical backbone can additionally project the representation to a configurable feature dimension.

---

# 2. Parallel Quantum Processing

The central idea of the architecture is the use of multiple independent quantum heads.

The default parallel configuration is:

```text
Number of quantum heads = 3
```

Each head contains its own QNode and independent trainable quantum parameters.

The outputs of all heads are concatenated:

```text
Head 1 ──┐
Head 2 ──┼──► Concatenate ──► Quantum Feature Vector
Head 3 ──┘
```

Each quantum head produces measurements from every qubit using:

```text
<X>
<Z>
```

Therefore, with:

```text
n_qubits = 6
n_parallel = 3
```

the resulting quantum feature dimension is:

```text
3 × 6 × 2 = 36
```

The architecture explicitly defines the quantum output dimension as:

```text
n_parallel × n_qubits × 2
```

before passing it to the final classifier.

---

# Quantum Encoding Strategies

The project investigates three main classical-to-quantum encoding modes.

---

## 3. Amplitude Embedding

Amplitude encoding maps a classical vector into the amplitudes of a quantum state.

For `N` qubits, the Hilbert space contains:

```text
2^N
```

basis states.

For example:

```text
N = 7

2^7 = 128
```

Therefore, the CIFAR-10 experiments use:

```text
feature_dim = 128
n_qubits = 7
```

for the amplitude representation.

Conceptually:

$$
|x\rangle =
\sum_{i=0}^{2^N-1} x_i |i\rangle
$$

with the feature vector normalized before state preparation.

The implementation uses PennyLane's:

```python
qml.AmplitudeEmbedding(...)
```

with normalization and zero padding enabled.

### Advantages

* Compact representation in Hilbert space
* `2^N` amplitudes represented using `N` qubits
* Natural fit for high-dimensional vectors

### Limitations

Amplitude encoding requires the input vector to match the state-vector dimensionality, making the preprocessing/projection stage important.

---

# 4. ZZ Feature Encoding

The second strategy uses angle-based encoding combined with data-dependent ZZ interactions.

For each qubit, two classical values are used:

```text
θy → RY
θz → RZ
```

The quantum input is therefore organized as:

```text
2 × n_qubits
```

per layer.

For the standard six-qubit configuration:

```text
2 × 6 = 12 features per layer
```

The circuit applies:

```text
Hadamard
   ↓
RY(θy)
   ↓
RZ(θz)
   ↓
Data-dependent ZZ interaction
```

The ZZ coupling is controlled by:

$$
\phi =
s\frac{\theta_{z,q}\theta_{z,j}}{\pi}
$$

where `s` is the configurable:

```text
zz_scale
```

parameter.

The interaction is implemented through a CNOT–RZ–CNOT construction.

---

# 5. Dense-Angle Encoding

The third encoding strategy uses three angles per qubit:

```text
θx
θy
θz
```

which are mapped to:

```text
RX(θx)
RY(θy)
RZ(θz)
```

Thus each layer consumes:

```text
3 × n_qubits
```

features.

For six qubits:

```text
3 × 6 = 18
```

features per layer.

The angles are normalized and passed through `tanh` before being mapped to the interval associated with rotation angles.

A data-dependent interaction is also constructed between neighboring qubits.

---

# Variational Quantum Ansatz

After data encoding, the quantum circuit applies a trainable variational block.

The core structure is:

```text
Rot
 │
 ▼
Entanglement
 │
 ▼
Rot
```

In the improved/imprinting configuration, the architecture can use a deeper pattern:

```text
Rot
 │
 ▼
Entanglement
 │
 ▼
Rot
 │
 ▼
Entanglement
 │
 ▼
Rot
```

The trainable rotation blocks use PennyLane's:

```python
qml.Rot(...)
```

with three trainable parameters per qubit.

The circuit is therefore both:

* **data dependent**, through the embedding layers
* **trainable**, through variational quantum parameters

The circuit uses the PyTorch interface:

```python
interface="torch"
```

and backpropagation:

```python
diff_method="backprop"
```

allowing the quantum parameters to participate in end-to-end optimization.

---

# Entanglement Topologies

A major component of the project is the comparison of different quantum connectivity patterns.

Three topologies are implemented.

## Ring

Each qubit interacts with its neighbor:

```text
0 → 1
1 → 2
2 → 3
...
N-1 → 0
```

This produces a circular connectivity structure.

---

## Pairwise

The circuit first connects neighboring qubits in pairs and then applies another layer of pairwise interactions.

Conceptually:

```text
0 ── 1
2 ── 3
4 ── 5

then

1 ── 2
3 ── 4
...
```

This creates a structured local interaction pattern.

---

## Ring-Skip

The ring-skip topology introduces both:

```text
q → q+1
```

and:

```text
q → q+2
```

connections.

This introduces longer-range interactions than the simple ring topology.

The three patterns are explicitly defined in the notebook and assigned to the parallel heads.

---

# Dynamic Entanglement Imprinting

The project also experiments with **dynamic entanglement routing**, referred to in the notebook as *imprinting*.

Instead of using exactly the same target wires at every variational layer, the target connectivity can shift according to:

$$
shift = layer \bmod N
$$

where `N` is the number of qubits.

Thus the connectivity changes as the circuit depth increases.

Conceptually:

```text
Layer 0 → shift = 0
Layer 1 → shift = 1
Layer 2 → shift = 2
...
```

The purpose is to alter the interaction routing while keeping the basic embedding and variational structure intact.

The notebook explicitly compares baseline routing against imprinting-shift routing.

---

# Parallel Head Assignment

The three quantum heads use the three entanglement patterns cyclically:

| Quantum Head | Entanglement |
| ------------ | ------------ |
| Head 0       | Ring         |
| Head 1       | Pairwise     |
| Head 2       | Ring-Skip    |

This creates architectural diversity across the parallel quantum paths.

The assignment is controlled by:

```python
ENTANGLEMENT_NAMES[head_idx % len(ENTANGLEMENT_NAMES)]
```

so the design can be extended to a larger number of parallel heads.

---

# Quantum Readout

The quantum circuits measure two observables per qubit:

$$
\langle X_q\rangle
$$

and

$$
\langle Z_q\rangle
$$

for every qubit.

Therefore:

```text
Measurements per qubit = 2
```

and:

```text
Quantum output =
n_parallel × n_qubits × 2
```

For the six-qubit, three-head configuration:

```text
3 × 6 × 2 = 36
```

The resulting vector is passed to a classical linear classifier.

---

# Complete Model Architecture

The full hybrid architecture can be summarized as:

```text
                     Input Image
                         │
                         ▼
              ┌─────────────────────┐
              │   Classical CNN     │
              │                     │
              │ Conv → BN → ReLU    │
              │ MaxPool             │
              │ Conv → BN → ReLU    │
              │ MaxPool             │
              └──────────┬──────────┘
                         │
                         ▼
                Classical Feature Map
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
           Head 0     Head 1     Head 2
           Ring       Pairwise   Ring-Skip
              │          │          │
              ▼          ▼          ▼
          Quantum     Quantum    Quantum
          Circuit     Circuit    Circuit
              │          │          │
              └──────────┼──────────┘
                         │
                         ▼
                  <X> + <Z>
                         │
                         ▼
              Concatenated Q Features
                         │
                         ▼
                Linear Classifier
                         │
                         ▼
                   10 Classes
```

---

# Implemented Models

The notebook benchmarks four models:

### 1. `HybridAmplitudeParallel`

Classical CNN + parallel quantum heads using amplitude embedding.

### 2. `HybridZZFeatureParallel`

Classical CNN + parallel quantum heads using ZZ-based feature encoding.

### 3. `HybridDenseAngleParallel`

Classical CNN + parallel quantum heads using dense-angle encoding.

### 4. `FullClassicalBaseline`

A purely classical baseline using the same classical backbone followed by:

```text
Linear(feature_dim → 64)
ReLU
Linear(64 → 32)
SELU
Linear(32 → num_classes)
```

The baseline is important because it provides a direct reference point for evaluating whether the hybrid quantum architecture improves classification performance under the same experimental setting.

---

# Dataset

The implementation supports:

* MNIST
* Fashion-MNIST
* CIFAR-10
* Custom image datasets using `ImageFolder`

The CIFAR-10 experiments are the main benchmark reported toward the end of the notebook.

For CIFAR-10, images are resized to:

```text
32 × 32
```

Training augmentation includes:

* Random horizontal flip
* Random crop with padding

Normalization uses the standard CIFAR-10 channel statistics:

```text
Mean = (0.4914, 0.4822, 0.4465)

Std  = (0.2023, 0.1994, 0.2010)
```

The test transformation performs resizing, tensor conversion, and normalization without the random training augmentations.

---

# CIFAR-10 Configuration

The final CIFAR-10 experiment uses:

| Parameter         |                               Value |
| ----------------- | ----------------------------------: |
| Dataset           |                            CIFAR-10 |
| Input size        |                           `32 × 32` |
| Validation ratio  |                              `0.10` |
| Batch size        |                                `64` |
| Epochs            |                                `18` |
| Learning rate     |                              `1e-3` |
| Weight decay      |                              `1e-3` |
| Feature dimension |                               `128` |
| Qubits            |                                 `7` |
| Quantum layers    | `4` in the experiment configuration |
| Parallel heads    |                                 `3` |
| Quantum device    |                     `default.qubit` |
| Save directory    | `./results_hybrid_parallel_cifar10` |

The notebook's CIFAR-10 configuration explicitly sets these parameters and stores the resulting configuration as `config.json`.

> **Note:** The final paper-style summary printed at the end reports `n_layers=3`, while the CIFAR-10 configuration cell shown earlier sets `n_q_layers=4`. This README intentionally preserves the values actually present in the notebook rather than silently resolving this inconsistency.

---

# Training

The training pipeline is implemented in PyTorch.

The loss function is:

```python
F.cross_entropy(logits, y)
```

The optimizer is:

```text
AdamW
```

with configurable:

```text
learning rate
weight decay
```

The default scheduler is:

```text
CosineAnnealingLR
```

with:

```text
eta_min = 0.01 × learning_rate
```

The training loop also supports:

* CUDA acceleration
* Automatic mixed precision when enabled
* Gradient scaling
* Gradient clipping
* Best-model checkpointing
* Early stopping

Gradient norms are clipped to:

```text
max_norm = 1.0
```

## The optimizer and scheduler factories are implemented directly in the notebook.

# Early Stopping

The training framework tracks validation accuracy and saves the best model.

An improvement is accepted when:

```text
val_acc > best_val_acc + min_delta
```

where:

```text
min_delta = 1e-4
```

The training terminates after a configurable number of consecutive epochs without improvement.

For the CIFAR-10 experiments, the notebook uses:

```text
patience = 5
```

The best model state is stored as:

```text
<model_name>_best.pth
```

inside:

```text
results_hybrid_parallel_cifar10/models/
```

The training implementation and checkpointing logic are shown in the notebook's `train_one_model` function.

---

# Evaluation Metrics

The project evaluates the models using:

### Accuracy

The fraction of correctly classified test examples.

### Macro F1

F1 averaged equally across all classes.

### Weighted F1

F1 weighted by the number of examples in each class.

### Cross-Entropy Loss

Used as the classification loss.

### Confusion Matrix

Used to inspect class-level classification behavior.

The evaluation routine also generates a full `classification_report`.

---

# Quantum Circuit Analysis

In addition to classification accuracy, the notebook studies properties of the quantum circuits themselves.

Two important metrics are calculated.

## Expressibility

The notebook estimates circuit expressibility by comparing the empirical fidelity distribution of generated quantum states with the Haar-random fidelity distribution.

The divergence is calculated using KL divergence:

```text
Expressibility (KL)
```

Lower values indicate a closer match to the reference distribution under the metric used in the notebook.

The calculation samples pairs of circuit states and computes:

$$
F(\psi_1,\psi_2)
=
|\langle\psi_1|\psi_2\rangle|^2
$$

before comparing the empirical distribution with the Haar distribution.

---

# Meyer–Wallach Entanglement

The notebook also calculates the Meyer–Wallach global entanglement measure.

For each qubit, a reduced density matrix is constructed using a partial trace, and its purity is used to estimate the entanglement contribution.

The implementation averages the resulting values over multiple random circuit configurations.

The notebook reports:

```text
Meyer-Wallach ↑
```

where larger values are treated as stronger global entanglement under the experiment's interpretation.

---

# Example Quantum Property Results

One of the recorded six-qubit experiments reports the following values:

| Mode        | Head | Topology  | Expressibility KL ↓ | Meyer–Wallach ↑ |
| ----------- | ---: | --------- | ------------------: | --------------: |
| Amplitude   |    0 | Ring      |              0.0047 |          0.9554 |
| Amplitude   |    1 | Pairwise  |              0.0244 |          0.9472 |
| Amplitude   |    2 | Ring-Skip |              0.0275 |          0.9569 |
| ZZ          |    0 | Ring      |              0.0325 |          0.9534 |
| ZZ          |    1 | Pairwise  |              0.0189 |          0.9222 |
| ZZ          |    2 | Ring-Skip |              0.0092 |          0.9548 |
| Dense-Angle |    0 | Ring      |              0.0210 |          0.9517 |
| Dense-Angle |    1 | Pairwise  |              0.0188 |          0.9288 |
| Dense-Angle |    2 | Ring-Skip |              0.0123 |          0.9552 |

These values come from an executed quantum-property evaluation in the notebook and should be interpreted as experiment-specific results rather than universal properties of the embedding methods.

---

# Recorded MNIST Results

The notebook also contains an MNIST experiment using the hybrid models and the classical baseline.

The recorded test results are:

| Model                    | Test Accuracy | Macro F1 | Weighted F1 | Test Loss |
| ------------------------ | ------------: | -------: | ----------: | --------: |
| HybridAmplitudeParallel  |    **98.35%** |   98.34% |      98.35% |    0.0591 |
| HybridZZFeatureParallel  |        98.28% |   98.28% |      98.28% |    0.0599 |
| HybridDenseAngleParallel |        98.22% |   98.22% |      98.22% |    0.0684 |
| FullClassicalBaseline    |        98.01% |   98.00% |      98.01% |    0.0638 |

The strongest recorded MNIST result is obtained by `HybridAmplitudeParallel` with a test accuracy of `0.9835`.

---

# CIFAR-10 Results

The notebook contains multiple CIFAR-10 runs/configurations.

One recorded test evaluation reports:

| Model                    | Test Accuracy |   Macro F1 | Weighted F1 | Test Loss |
| ------------------------ | ------------: | ---------: | ----------: | --------: |
| HybridAmplitudeParallel  |        62.12% |     61.83% |      61.83% |    1.0859 |
| HybridZZFeatureParallel  |        61.72% |     61.16% |      61.16% |    1.1104 |
| HybridDenseAngleParallel |        60.04% |     59.62% |      59.62% |    1.1678 |
| FullClassicalBaseline    |    **64.18%** | **64.04%** |  **64.04%** |    0.9945 |

These values are explicitly printed during the CIFAR-10 test evaluation.

However, the notebook subsequently produces a separate **paper-ready final summary** for another CIFAR-10 configuration:

| Model                   | Test Accuracy |   Macro F1 | Weighted F1 | Test Loss | Best Val Accuracy | Epochs |
| ----------------------- | ------------: | ---------: | ----------: | --------: | ----------------: | -----: |
| HybridAmplitudeParallel |    **72.36%** | **72.11%** |  **72.11%** |    0.7965 |            69.42% |     15 |
| FullClassicalBaseline   |    **73.27%** | **73.14%** |  **73.14%** |    0.7605 |            70.04% |     15 |

The final summary identifies this experiment as:

```text
Dataset: CIFAR10
n_qubits: 7
n_layers: 3
n_parallel: 3
feature_dim: 128
```

and reports that the generated figures are stored under:

```text
results_hybrid_parallel_cifar10/figures
```

with circuit diagrams and architecture schematics in their respective subdirectories.

### Interpretation

In the final paper-style CIFAR-10 summary, the classical baseline remains slightly ahead:

```text
Classical baseline : 73.27%
Parallel amplitude : 72.36%
```

The gap is approximately:

```text
0.91 percentage points
```

Therefore, the results in this notebook should **not** be presented as evidence that the hybrid model universally outperforms the classical baseline.

Instead, they demonstrate that a parallel quantum-classical model can achieve competitive performance while using a substantially different computational architecture.

---

# Trainable Parameter Comparison

The notebook reports the following trainable parameter counts:

| Model                    | Trainable Parameters |
| ------------------------ | -------------------: |
| HybridAmplitudeParallel  |               18,518 |
| HybridZZFeatureParallel  |               18,582 |
| HybridDenseAngleParallel |               18,582 |
| FullClassicalBaseline    |               24,458 |

The hybrid models therefore use fewer trainable parameters than the classical baseline in the reported architecture.

For example:

```text
HybridAmplitudeParallel
18,518 parameters

FullClassicalBaseline
24,458 parameters
```

This corresponds to approximately **24% fewer trainable parameters** for the amplitude-based hybrid architecture relative to the reported classical baseline.

Parameter count alone, however, should not be interpreted as a direct measurement of computational cost because quantum-circuit simulation and quantum hardware execution introduce different costs and bottlenecks.

---

# Computational Considerations

The quantum components are simulated using PennyLane's:

```python
qml.device("default.qubit", ...)
```

This means the experiments in the notebook are **quantum-circuit simulations**, not executions on physical quantum hardware.

The computational cost can become substantial as:

* number of qubits increases
* circuit depth increases
* number of parallel heads increases
* number of training batches increases

This is especially visible in the CIFAR-10 experiments.

For example, the recorded CIFAR-10 training runs show that quantum models can require substantially more time per epoch than the purely classical baseline.
Therefore, the term **parallel** in this project refers primarily to the architectural organization into multiple quantum heads; it should not automatically be interpreted as simultaneous execution on independent physical quantum processors.

---

# Reproducibility

The project explicitly uses deterministic seeds in multiple parts of the experiment.

The configuration defines:

```python
seed = 43
```

and dataset splitting uses a seeded PyTorch generator.

The notebook also provides:

```python
set_seed(42)
```

for:

* PyTorch
* NumPy
* Python's `random`

This is intended to improve reproducibility of dataset sampling and model initialization.
Because quantum simulation, numerical libraries, hardware backends, and multiprocessing can introduce additional sources of nondeterminism, exact numerical reproduction may still depend on the software and execution environment.

---

# Installation

## Requirements

The notebook imports the following major packages:

```text
Python 3.11
PyTorch
TorchVision
PennyLane
NumPy
SciPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Jupyter
```

The notebook metadata indicates:

```text
Python 3.11.2
```

as the recorded Python version.

A minimal installation can be performed with:

```bash
pip install torch torchvision pennylane numpy scipy pandas scikit-learn matplotlib seaborn jupyter
```

---

# Running the Project

Clone the repository:

```bash
git clone https://github.com/ammar31300/A-Parallel-Quantum-Classical-Hybrid-Architecture-for-Image-Classification.git
cd A-Parallel-Quantum-Classical-Hybrid-Architecture-for-Image-Classification
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
paralell-hybridmoded.ipynb
```

Run the notebook sequentially from the configuration and dataset sections through model construction, training, evaluation, and visualization.

---

# Quick Start

For a basic experiment:

```python
cfg = ExperimentConfig()

train_loader, val_loader, test_loader, in_channels, num_classes = \
    DatasetFactory.build_loaders(cfg)
```

Create a hybrid model:

```python
model = HybridAmplitudeParallelModel(
    in_channels=in_channels,
    cfg=cfg
)
```

Create an optimizer:

```python
optimizer = OptimizerFactory.build_optimizer(
    model,
    cfg
)
```

Create a scheduler:

```python
scheduler = OptimizerFactory.build_scheduler(
    optimizer,
    cfg
)
```

Then train:

```python
history, best_state = train_one_model(
    name="HybridAmplitudeParallel",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    cfg=cfg,
    device=DEVICE
)
```

Finally evaluate the best checkpoint:

```python
test_result = evaluate_one_model_on_test(
    name="HybridAmplitudeParallel",
    model=model,
    best_state=best_state,
    test_loader=test_loader,
    cfg=cfg,
    device=DEVICE
)
```

---

# Custom Dataset

The implementation supports custom datasets through `torchvision.datasets.ImageFolder`.

Set:

```python
cfg.dataset_name = "custom"

cfg.custom_train_dir = "path/to/train"
cfg.custom_test_dir = "path/to/test"
```

The directory structure should follow the standard `ImageFolder` format:

```text
dataset/
├── train/
│   ├── class_0/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
│
└── test/
    ├── class_0/
    ├── class_1/
    └── ...
```

The dataset factory automatically determines the number of classes when the underlying dataset exposes a `classes` attribute.

---

# Project Structure

A conceptual structure of the repository is:

```text
A-Parallel-Quantum-Classical-Hybrid-Architecture-for-Image-Classification/
│
├── paralell-hybridmoded.ipynb
│
├── CNN1.ipynb
├── CNN4.ipynb
├── Hybrid_HQNN.ipynb
├── Qounvolution.ipynb
├── SeekThermal_parallel_HybridModel.ipynb
│
├── data/
│
├── results_hybrid_parallel/
│
├── results_hybrid_parallel_cifar10/
│   ├── models/
│   ├── figures/
│   │   ├── circuits/
│   │   ├── architecture/
│   │   └── embeddings/
│   └── reports/
│
└── README.md
```

The exact contents may vary depending on which experiments have been executed and which generated artifacts have been committed.

The main notebook automatically creates directories for:

```text
models/
figures/
reports/
```

and writes the experiment configuration to:

```text
config.json
```

---

# Generated Figures

The notebook includes dedicated functions for generating publication-ready visualizations.

These include:

## Circuit diagrams

```text
figures/circuits/
```

## Embedding diagrams

```text
figures/embeddings/
```

## Architecture schematic

```text
figures/architecture/
```

## Training convergence

The visualization pipeline generates:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

## Benchmark comparison

The notebook generates model comparison plots for:

* Best training accuracy
* Final training accuracy
* Best validation accuracy
* Test accuracy

## Generalization gap

The notebook also computes and visualizes differences between training, validation, and test performance.

## Confusion matrices

A modular confusion-matrix visualization is generated for the evaluated models.

## The visualization functions explicitly save high-resolution PNG/PDF figures and architecture/circuit diagrams.

# Dimension Map

For the final CIFAR-10 architecture reported by the notebook:

```text
Input
(B, 3, 32, 32)

        │
        ▼

Classical CNN
Feature representation
(B, 128)

        │
        ├──────────────┐
        │              │
        ▼              ▼
 Quantum Head 0    Quantum Head 1 ... Head 2

        │
        ▼

Per-head measurements
<X>, <Z>

        │
        ▼

Concatenated quantum representation

        │
        ▼

Linear classifier

        │
        ▼

10 output classes
```

The notebook's circuit-generation utilities explicitly document the dimensional flow from the input image through the classical representation, per-head quantum inputs, measurements, and classifier.

---

# Experimental Design

The experiments investigate several dimensions simultaneously:

### Classical representation

How effectively can a compact CNN representation be transformed into quantum-compatible features?

### Quantum embedding

How do:

```text
Amplitude
ZZ
Dense-Angle
```

encodings behave?

### Entanglement topology

How do:

```text
Ring
Pairwise
Ring-Skip
```

connectivity patterns affect the circuit?

### Dynamic routing

Does changing entanglement routing between layers affect circuit properties?

### Parallelization

Can multiple quantum heads provide a richer quantum representation than a single quantum path?

### Classical comparison

How does the hybrid architecture compare against a classical model using the same backbone?

---

# Important Implementation Details

## Amplitude mode

Amplitude mode operates directly on the spatial feature map.

The feature map is permuted independently for each head before pooling and projection.

Different heads therefore see different deterministic spatial orderings.

The project implements multiple permutation families, including:

* Identity
* Serpentine
* Checkerboard
* Row-shift
* Column-shift
* Reverse
* Affine
* Diagonal-wrap

This provides head-specific spatial transformations before amplitude encoding.

---

## ZZ and Dense-Angle modes

The feature vector is reshaped into head/layer-specific quantum inputs.

If the classical representation is larger than the required quantum input dimension, a trainable projection is used.

If it is smaller, the implementation uses cyclic feature repetition.

## This makes the quantum input pipeline flexible with respect to the configured feature dimension.

# Limitations

Several limitations should be considered when interpreting the results.

## 1. Quantum simulation

The reported experiments use a simulated quantum device:

```text
default.qubit
```

and therefore do not demonstrate performance on actual quantum hardware.

## 2. Computational cost

Quantum circuit simulation becomes expensive as the number of qubits, circuit depth, and number of circuit evaluations increases.

## 3. Limited CIFAR-10 performance

In the final paper-style result available in the notebook, the classical baseline achieves slightly higher test accuracy than the amplitude-based hybrid model.

Therefore, the current experiments do not establish a universal classification advantage for the hybrid approach.

## 4. Multiple experimental configurations

The notebook contains several experimental stages and configuration changes.

For rigorous scientific reporting, the configuration associated with each reported result should always be recorded.

## 5. Reproducibility across environments

Results can vary depending on:

* PennyLane version
* PyTorch version
* NumPy/SciPy version
* CPU/GPU environment
* random seeds
* simulator behavior

---

# Future Work

Potential extensions of this architecture include:

* Execution on real quantum hardware
* Noise-aware training
* Hardware-efficient ansätze
* More efficient amplitude/state preparation
* Larger quantum-head ensembles
* Learnable entanglement topologies
* Adaptive head routing
* Quantum attention mechanisms
* Quantum convolutional layers
* Hybrid residual architectures
* Larger image datasets
* ImageNet-scale experiments
* Transfer learning
* Parameter-efficient quantum adapters
* Noise robustness evaluation
* Hardware-aware circuit optimization
* Comparison with stronger classical CNN baselines
* Statistical significance testing over multiple random seeds

---

# Scientific Interpretation

The central contribution of this implementation is not simply replacing a classical layer with a quantum circuit.

Instead, it explores a **parallel hybrid representation-learning strategy** in which:

1. classical convolution extracts spatial features;
2. different quantum heads transform those features using different quantum encodings;
3. each head employs an independent trainable variational circuit;
4. different entanglement structures provide architectural diversity;
5. measured quantum observables form a compact learned representation;
6. a classical classifier performs the final decision.

This makes the architecture particularly useful as an experimental framework for studying the interaction between:

```text
Classical representation learning
            +
Quantum feature transformation
            +
Parallel quantum processing
            +
Entanglement topology
```

---

# Reproducible Research Checklist

For reproducing a reported experiment, record at least:

```text
Dataset
Dataset fraction
Train/validation split
Random seed
Input resolution
Feature dimension
Number of qubits
Number of quantum layers
Number of parallel heads
Embedding mode
Entanglement topology
ZZ scaling factor
Optimizer
Learning rate
Weight decay
Scheduler
Batch size
Number of epochs
Early-stopping patience
Quantum backend
```

This is especially important because changing the number of qubits or feature dimension changes the quantum representation itself.

---

# Citation

If you use this repository or architecture in academic work, please cite the repository:

```bibtex
@misc{ammar_parallel_quantum_classical_hybrid,
  author       = {Ammar},
  title        = {A Parallel Quantum-Classical Hybrid Architecture for Image Classification},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/ammar31300/A-Parallel-Quantum-Classical-Hybrid-Architecture-for-Image-Classification}
}
```

Please update the author/year metadata if a formal publication associated with this work becomes available.

---

# Acknowledgments

This project makes use of the following open-source technologies:

* PyTorch
* TorchVision
* PennyLane
* NumPy
* SciPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter

These tools provide the classical deep-learning, quantum-machine-learning, numerical, evaluation, and visualization components required by the experiments.

---

# License

Please refer to the repository's license file for the applicable terms of use.

If no license has been added to the repository, users should not assume that the code is freely licensed for redistribution or commercial use.

---

# Repository

**GitHub:**
https://github.com/ammar31300/A-Parallel-Quantum-Classical-Hybrid-Architecture-for-Image-Classification

**Main Notebook:**
`paralell-hybridmoded.ipynb`

---

## Summary

This repository presents a modular framework for investigating **parallel quantum-classical hybrid neural networks for image classification**.

The architecture combines:

```text
CNN Feature Extraction
        +
Multiple Quantum Processing Heads
        +
Amplitude / ZZ / Dense-Angle Encoding
        +
Dynamic Entanglement Topologies
        +
Variational Quantum Circuits
        +
<X>/<Z> Quantum Readout
        +
Classical Classification
```

The experiments demonstrate that the proposed hybrid models can achieve competitive image-classification performance, while also providing a framework for studying quantum-circuit expressibility, entanglement, embedding strategies, and parameter efficiency.

The current results should be interpreted as an experimental study of hybrid quantum-classical architectures rather than as evidence that quantum models universally outperform classical neural networks.
