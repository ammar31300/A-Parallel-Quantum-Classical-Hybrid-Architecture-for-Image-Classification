# A Parallel Quantum-Classical Hybrid Architecture for Image Classification

<p align="center">

**A parallel hybrid quantum-classical framework for image classification using convolutional feature extraction and multiple parameterized quantum neural network branches.**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20ML-6c5ce7)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Quantum](https://img.shields.io/badge/Quantum-Hybrid-purple)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 📌 Overview

This repository presents a **Parallel Quantum-Classical Hybrid Architecture** for image classification.

The central idea is to combine the strengths of classical convolutional neural networks with multiple parallel quantum neural network (QNN) branches.

Instead of sending classical CNN features into a single quantum circuit, the proposed architecture creates several **parallel quantum heads**, where each head receives a different representation or encoding of the extracted features.

The general pipeline is:

```text
Input Image
     │
     ▼
┌──────────────────────┐
│ Classical CNN        │
│ Feature Extraction   │
└──────────┬───────────┘
           │
           ▼
   Classical Features
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
 Quantum Head 1  Quantum Head 2  ... Quantum Head N
     │           │                    │
     └─────┬─────┴──────────┬─────────┘
           │
           ▼
     Quantum Measurements
           │
           ▼
   Concatenated Quantum
       Feature Vector
           │
           ▼
      Linear Classifier
           │
           ▼
        Prediction
```

The repository evaluates this concept on more than one image-classification setting, including a thermal-image dataset, while keeping the main parallel hybrid architecture consistent.

---

# 🎯 Motivation

Classical CNNs are highly effective at extracting spatial features from images. However, quantum machine learning provides an alternative mechanism for nonlinear feature transformation through parameterized quantum circuits.

A hybrid architecture attempts to combine these two paradigms:

* **CNNs** provide efficient spatial feature extraction.
* **Quantum circuits** provide nonlinear quantum transformations.
* **Parallel quantum branches** provide multiple feature-processing pathways.
* **Classical classification layers** convert the resulting quantum features into final predictions.

The goal is not to claim an automatic quantum advantage, but to investigate whether a parallel quantum-classical architecture can provide a useful and experimentally scalable framework for image classification.

---

# 🧠 Main Contributions

The implementation focuses on several ideas.

### 1. Classical feature extraction

A convolutional backbone first converts the input image into a compact feature representation.

### 2. Parallel quantum processing

Instead of using one quantum circuit, multiple independent quantum heads are executed in parallel conceptually:

```text
Feature Representation
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
QNN-1  QNN-2  QNN-3
 │      │      │
 └──────┼──────┘
        ▼
Concatenated Quantum Features
```

### 3. Multiple quantum encoding strategies

The implementation supports different quantum feature-processing modes:

* Amplitude encoding
* ZZ-based feature encoding
* Dense angle encoding

### 4. Different entanglement patterns

The quantum circuits can use different entanglement structures, including:

* Ring entanglement
* Pairwise entanglement
* Ring + skip connections

### 5. Classical baseline

A purely classical model is trained under the same general experimental framework to provide a reference point.

---

# 🏗️ Architecture

## High-Level Architecture

The complete hybrid architecture can be summarized as:

```text
                  Input Image
                       │
                       ▼
             ┌─────────────────┐
             │ Classical CNN   │
             │    Backbone     │
             └────────┬────────┘
                      │
              Feature Representation
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
     Quantum Head  Quantum Head  Quantum Head
         #1            #2            #3
          │           │           │
          ▼           ▼           ▼
      Quantum       Quantum       Quantum
    Measurement   Measurement   Measurement
          │           │           │
          └───────────┼───────────┘
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

---

# 🔬 Classical CNN Backbone

The classical backbone is responsible for extracting spatial representations from the input image.

The implemented convolutional encoder contains four convolutional stages:

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
      BatchNorm
      ReLU
```

The channel progression is:

```text
Input channels
      ↓
16 channels
      ↓
32 channels
      ↓
64 channels
      ↓
64 channels
```

The backbone can return both:

* a compact feature vector
* a spatial feature map

This is particularly important for the amplitude-based parallel quantum branch.

---

# ⚛️ Parallel Quantum Neural Network

The main quantum component is implemented through a reusable `BaseParallelQNNLayer`.

Conceptually, it performs:

```text
Classical Features
       │
       ▼
Head-specific preparation
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
QNode  QNode  QNode
 1      2      3
 │      │      │
 └──────┼──────┘
        ▼
 Concatenation
        │
        ▼
Quantum Feature Vector
```

Each quantum head owns:

* an independent QNode
* trainable variational parameters
* a head-specific feature representation
* a potentially different entanglement pattern

For each head, the trainable parameter tensor follows the structure:

```text
(n_layers, 2, n_qubits, 3)
```

where the two blocks correspond to two parameterized rotation stages.

---

# 🔀 Parallel Head Design

The default SeekThermal configuration uses:

```text
Number of parallel heads = 3
Number of qubits         = 6
Quantum layers           = 3
```

Therefore each quantum head produces measurements from six qubits.

For every qubit, the implementation measures:

```text
⟨X⟩
⟨Z⟩
```

Consequently, each head produces:

```text
6 qubits × 2 observables = 12 features
```

With three parallel heads:

```text
3 × 6 × 2 = 36 quantum features
```

These features are concatenated and passed to the final classifier.

---

# ⚛️ Quantum Encoding Modes

The repository implements three major quantum variants.

---

## 1. HybridAmplitudeParallel

### Amplitude Encoding

The amplitude variant uses the CNN spatial feature map and converts it into a vector suitable for amplitude encoding.

The implementation applies:

```text
CNN Feature Map
      │
      ▼
Spatial permutation
      │
      ▼
Adaptive Average Pooling
      │
      ▼
Flatten
      │
      ▼
Linear projection
      │
      ▼
Amplitude vector
      │
      ▼
AmplitudeEmbedding
```

For six qubits, the Hilbert-space dimension is:

```text
2^6 = 64
```

The configuration therefore uses a 64-dimensional quantum representation.

The circuit initializes the quantum state using:

```python
qml.AmplitudeEmbedding(
    features=amp_batch,
    wires=range(n_qubits),
    normalize=True,
    pad_with=0.0,
)
```

---

# 🔢 Spatial Permutation Strategy

One interesting part of the amplitude model is the use of different deterministic spatial permutations for different quantum heads.

The implementation supports permutation families such as:

* Identity
* Serpentine
* Checkerboard
* Row-shift
* Column-shift
* Reverse
* Affine permutation
* Diagonal wrapping

This allows different quantum heads to process different spatial arrangements of the same CNN feature map.

Conceptually:

```text
CNN Feature Map
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
Perm1 Perm2 Perm3
 │     │     │
 ▼     ▼     ▼
QNN1  QNN2  QNN3
```

This provides diversity between the parallel branches without requiring completely different CNN backbones.

---

# 2. HybridDenseAngleParallel

The dense-angle variant converts the classical feature vector into quantum rotation angles.

The feature vector is reshaped into:

```text
Parallel Heads
      ×
Quantum Layers
      ×
2 × Number of Qubits
```

The implementation applies normalization and maps the resulting values using:

```python
torch.tanh(x) * torch.pi
```

The quantum circuit then applies parameterized rotations such as:

```text
RX
RY
```

followed by data-dependent interactions and variational layers.

The overall structure is:

```text
Classical Features
       │
       ▼
Feature Projection / Reshaping
       │
       ▼
Angle Encoding
       │
       ▼
RX + RY
       │
       ▼
Data-dependent ZZ interaction
       │
       ▼
Variational Rot
       │
       ▼
Entanglement
       │
       ▼
Variational Rot
       │
       ▼
Measurement
```

---

# 3. HybridZZFeatureParallel

The ZZ-based architecture uses two feature channels for every qubit.

For each quantum layer:

```text
Feature Channel 1 → RY
Feature Channel 2 → RZ
```

The implementation also introduces a data-dependent ZZ interaction:

```text
CNOT
  │
 RZ
  │
CNOT
```

The interaction strength is controlled by:

```python
zz_scale = 0.25
```

The circuit therefore combines:

* single-qubit data encoding
* nonlinear interaction
* variational rotations
* entanglement
* quantum measurement

---

# 🔗 Variational Quantum Circuit

All three quantum modes share a common variational structure.

Each quantum layer follows approximately:

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

The trainable rotation is implemented with:

```python
qml.Rot(
    theta,
    phi,
    omega,
    wires=q
)
```

This gives each qubit three trainable rotational parameters per rotation block.

---

# 🔗 Entanglement Patterns

Three entanglement patterns are implemented.

## Ring

Each qubit interacts with the next qubit:

```text
q0 ──► q1
      │
q2 ◄──q1 ...
```

More precisely, the circuit creates cyclic nearest-neighbor interactions.

---

## Pairwise

The circuit creates alternating pairwise interactions.

Conceptually:

```text
q0 ── q1

q2 ── q3

q4 ── q5
```

followed by shifted interactions.

---

## Ring + Skip

This pattern combines:

* nearest-neighbor interactions
* longer-range skip interactions

This increases connectivity while keeping the circuit relatively structured.

---

# 🔄 Dynamic Entanglement Routing

The entanglement pattern receives a layer-dependent shift:

```python
current_shift = layer % n_qubits
```

Therefore, different quantum layers can use different interaction connectivity.

This provides a simple form of dynamic circuit routing while maintaining a relatively shallow architecture.

---

# 📏 Quantum Measurements

The final quantum representation is obtained by measuring two observables for every qubit:

```text
Pauli-X expectation
Pauli-Z expectation
```

The circuit returns:

```python
[
    <X(q0)>,
    <X(q1)>,
    ...
    <X(qN)>,
    <Z(q0)>,
    <Z(q1)>,
    ...
    <Z(qN)>
]
```

For:

```text
6 qubits
2 observables
3 parallel heads
```

the resulting feature size is:

```text
6 × 2 × 3 = 36
```

These 36 quantum features are fed into the final classifier.

---

# 🧮 Final Classifier

The hybrid models use a linear classifier:

```text
Quantum Features
      │
      ▼
Linear(36 → Number of Classes)
      │
      ▼
Class Logits
```

For the SeekThermal experiment:

```text
36 quantum features
        ↓
3 output classes
```

The classes are:

```text
Car
Cat
Man
```

---

# 🆚 Classical Baseline

To determine whether the quantum component provides useful representational behavior, the repository also implements:

```text
FullClassicalBaseline
```

The baseline uses the same classical backbone and replaces the quantum module with a classical multilayer classifier.

Its classifier is:

```text
Feature Vector
     │
     ▼
Linear
     │
     ▼
ReLU
     │
     ▼
Linear
     │
     ▼
SELU
     │
     ▼
Linear
     │
     ▼
Class Prediction
```

This provides a direct classical reference against which the hybrid variants can be evaluated.

---

# 📊 Experimental Variants

The complete experimental comparison is:

| Model                      | Classical CNN |      Quantum Module | Parallel Heads |
| -------------------------- | ------------: | ------------------: | -------------: |
| `HybridAmplitudeParallel`  |             ✓ |           Amplitude |              3 |
| `HybridDenseAngleParallel` |             ✓ |         Dense Angle |              3 |
| `HybridZZFeatureParallel`  |             ✓ | ZZ Feature Encoding |              3 |
| `FullClassicalBaseline`    |             ✓ |                None |              0 |

---

# 🌡️ SeekThermal Dataset Experiment

One of the experiments applies the architecture to the **SeekThermal** thermal image dataset.

The task is a three-class object classification problem:

```text
Car
Cat
Man
```

The dataset is organized into training and testing directories:

```text
SeekThermal/
├── Train/
│   ├── Car/
│   ├── Cat/
│   └── Man/
│
└── Test/
    ├── Car/
    ├── Cat/
    └── Man/
```

---

# 🖼️ Image Preprocessing

Images are resized to:

```text
128 × 96
```

and converted into tensors.

The normalization is:

```python
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

Therefore, the preprocessing pipeline is:

```text
Original Thermal Image
        │
        ▼
Resize(128 × 96)
        │
        ▼
ToTensor()
        │
        ▼
Normalize(mean=0.5, std=0.5)
        │
        ▼
CNN
```

---

# 🧪 Train / Validation / Test Split

The training directory is divided into training and validation subsets.

The validation ratio is:

```text
15%
```

The split is stratified using the class labels.

Conceptually:

```text
Train Directory
      │
      ├──────── 85% → Training
      │
      └──────── 15% → Validation
```

The test dataset is loaded independently from the `Test` directory.

---

# ⚠️ Label Mapping

During the SeekThermal experiment, the notebook identifies a mismatch between the folder names and the semantic contents of the `Car` and `Cat` directories.

The training/validation labels are therefore explicitly remapped:

```text
Original:
Car → 0
Cat → 1
Man → 2

Corrected:
Car → 0
Cat → 1
Man → 2
```

with the underlying `Car ↔ Cat` sample-label assignment corrected where required.

The notebook verifies that the resulting class mapping is consistent across training, validation and test datasets.

This correction is important for reproducibility and should remain documented rather than silently removed.

---

# ⚙️ SeekThermal Configuration

The main configuration used in the provided experiment is:

| Parameter         |            Value |
| ----------------- | ---------------: |
| Number of classes |                3 |
| Classes           |  Car / Cat / Man |
| Image height      |              128 |
| Image width       |               96 |
| Batch size        |               64 |
| Epochs            |               10 |
| Learning rate     |             1e-3 |
| Weight decay      |             1e-4 |
| Optimizer         |            AdamW |
| Scheduler         | Cosine Annealing |
| Number of qubits  |                6 |
| Quantum layers    |                3 |
| Parallel heads    |                3 |
| Quantum device    |  `default.qubit` |
| Shots             |           `None` |
| ZZ scale          |             0.25 |
| Feature dimension |               64 |
| AMP               |         Disabled |

> The notebook contains a few configuration comments/assignments that differ from the final `ExperimentConfig`; the table above follows the effective `ExperimentConfig` values used by the training pipeline.

---

# ⚛️ Quantum Simulator

The experiments use PennyLane with:

```python
qml.device(
    "default.qubit",
    wires=n_qubits,
    shots=None
)
```

Therefore, the reported experiments are performed using a **state-vector quantum simulator**, not a physical quantum processor.

This distinction is important:

> The project demonstrates hybrid quantum-classical machine learning using quantum simulation. It should not be interpreted as demonstrating quantum computational speedup.

---

# 🏋️ Training

The training pipeline uses PyTorch and supports:

* Adam
* AdamW
* Cosine Annealing learning-rate scheduling
* Automatic mixed precision when supported
* Gradient clipping
* Early stopping
* Best-model checkpointing

The default optimizer is:

```text
AdamW
```

with:

```text
Learning rate = 0.001
Weight decay  = 0.0001
```

---

# ⏹️ Early Stopping

The training procedure monitors validation accuracy.

When the validation accuracy does not improve for a predefined number of epochs, training stops early.

The best model state is saved:

```text
{name}_best.pth
```

This helps prevent unnecessary training and preserves the best-performing model rather than simply using the final epoch.

---

# 📈 Evaluation Metrics

The models are evaluated using:

### Accuracy

```text
Correct predictions / Total predictions
```

### Macro F1

F1 is calculated independently for each class and then averaged equally.

This is particularly useful when class distributions are not perfectly balanced.

### Weighted F1

F1 scores are weighted according to class support.

### Confusion Matrix

The confusion matrix provides class-level information about:

```text
Car → Car
Car → Cat
Car → Man

Cat → Car
Cat → Cat
Cat → Man

Man → Car
Man → Cat
Man → Man
```

### Classification Report

The notebook also generates a complete classification report containing per-class:

* precision
* recall
* F1-score
* support

---

# 📊 Results

The experimental pipeline automatically stores the training histories and test results.

The following models are evaluated:

```text
HybridAmplitudeParallel
HybridZZFeatureParallel
HybridDenseAngleParallel
FullClassicalBaseline
```

The final comparison is intended to have the following form:

| Model                    | Test Accuracy | Macro F1 | Weighted F1 | Loss |
| ------------------------ | ------------: | -------: | ----------: | ---: |
| HybridAmplitudeParallel  |             — |        — |           — |    — |
| HybridZZFeatureParallel  |             — |        — |           — |    — |
| HybridDenseAngleParallel |             — |        — |           — |    — |
| FullClassicalBaseline    |             — |        — |           — |    — |

The notebook saves these results to:

```text
reports/
├── all_histories.json
└── all_test_results.json
```

The exact numerical values should be populated from the generated JSON files after the experiment has been executed.

---

# 📉 Training Curves

The project also generates scientific-style figures for:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy
* Validation Macro-F1
* Validation Weighted-F1
* Convergence behavior
* Model comparison

Figures are stored under the experiment output directory:

```text
figures/
└── test_XXX/
```

---

# 📁 Output Structure

A typical experiment generates an output structure similar to:

```text
results_hybrid_SeekThermal_parallel/
│
├── models/
│   ├── HybridAmplitudeParallel_best.pth
│   ├── HybridZZFeatureParallel_best.pth
│   ├── HybridDenseAngleParallel_best.pth
│   └── FullClassicalBaseline_best.pth
│
├── reports/
│   ├── all_histories.json
│   └── all_test_results.json
│
└── figures/
    └── test_XXX/
        └── ...
```

---

# 🧪 Reproducibility

A fixed random seed is used:

```python
SEED = 42
```

The experiment seeds:

* Python random
* NumPy
* PyTorch
* CUDA

and configures cuDNN for deterministic behavior.

This improves reproducibility, although exact results can still depend on the software/hardware environment.

---

# 💻 Installation

Create a Python environment:

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

Install the main dependencies:

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

---

# ▶️ Running the Experiment

Launch Jupyter:

```bash
jupyter notebook
```

Open the relevant notebook:

```text
paralell-hybridmoded.ipynb
```

or the SeekThermal experiment:

```text
SeekThermal_parallel_HybridModel.ipynb
```

Update the dataset path in the configuration:

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
3. Create stratified train/validation split
        │
        ▼
4. Apply image preprocessing
        │
        ▼
5. Build DataLoaders
        │
        ▼
6. Build CNN backbone
        │
        ▼
7. Build parallel quantum models
        │
        ▼
8. Build classical baseline
        │
        ▼
9. Train each model
        │
        ▼
10. Apply early stopping
        │
        ▼
11. Restore best validation model
        │
        ▼
12. Evaluate on test set
        │
        ▼
13. Generate classification reports
        │
        ▼
14. Generate confusion matrices
        │
        ▼
15. Save JSON results
        │
        ▼
16. Generate scientific plots
```

---

# 🧩 Repository Experiments

The repository is designed around a common architectural concept while allowing different datasets and configurations.

The major experimental direction is:

```text
                 Parallel Hybrid Architecture
                           │
          ┌────────────────┴────────────────┐
          │                                 │
    Image Dataset A                   SeekThermal
          │                                 │
          ▼                                 ▼
   Same Core Architecture            Same Core Architecture
          │                                 │
          └───────────────┬─────────────────┘
                          │
                          ▼
              Model Comparison
```

This makes the project suitable for studying whether the parallel quantum-classical design remains useful when the image domain changes.

---

# 🔍 Why Parallel Quantum Heads?

A single quantum circuit receives only one transformed representation of the classical features.

The parallel design instead creates multiple pathways:

```text
                     CNN Features
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   Representation 1  Representation 2  Representation 3
        │                 │                 │
        ▼                 ▼                 ▼
      QNN 1             QNN 2             QNN 3
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                    Feature Fusion
```

The intended benefit is increased representational diversity without simply making one quantum circuit deeper.

This is particularly relevant because increasing PQC depth can make optimization more difficult and can increase circuit execution cost.

---

# ⚖️ Hybrid vs Classical Learning

The project should be interpreted as an experimental comparison rather than a claim that quantum models universally outperform classical CNNs.

The classical baseline answers:

> How well can the same image-classification task be solved using only the classical representation?

The hybrid models answer:

> What changes when the learned representation is processed through parallel parameterized quantum circuits?

The comparison should therefore consider more than accuracy alone:

```text
Accuracy
Macro-F1
Weighted-F1
Loss
Parameter count
Training time
Inference cost
Convergence
```

---

# 🚧 Limitations

Several limitations should be considered.

## 1. Quantum simulation cost

The current implementation uses:

```text
PennyLane default.qubit
```

on a classical computer.

Quantum simulation can become expensive as the number of qubits and circuit evaluations increase.

---

## 2. No demonstrated quantum speedup

Using a quantum simulator does not establish quantum computational advantage.

The purpose of this repository is to investigate a hybrid architecture and its behavior.

---

## 3. Small quantum width

The SeekThermal configuration uses:

```text
6 qubits
```

This is intentionally manageable for simulation.

---

## 4. Dataset-specific preprocessing

The SeekThermal experiment uses specific image dimensions and class mappings.

When applying the architecture to another dataset, the following must be reconsidered:

* input channels
* image dimensions
* number of classes
* normalization
* feature dimension
* quantum encoding requirements

---

# 🔮 Future Work

Potential future improvements include:

### Real Quantum Hardware

Replace:

```text
default.qubit
```

with an actual quantum backend.

### Larger Quantum Circuits

Investigate:

```text
8+ qubits
```

while measuring the effect on performance and computational cost.

### Noise-Aware Training

Evaluate:

* bit-flip noise
* depolarizing noise
* readout noise
* gate noise

### Automated Quantum Architecture Search

Automatically search over:

* qubit count
* circuit depth
* encoding method
* entanglement topology
* observable selection

### Better Multi-Head Fusion

Instead of simple concatenation, investigate:

```text
Attention
Gating
Learnable fusion
Weighted averaging
Quantum-classical feature fusion
```

### More Classical Baselines

Future experiments could include:

* ResNet
* EfficientNet
* MobileNet
* Vision Transformer
* lightweight CNNs

to provide stronger classical reference points.

---

# 📚 Scientific Context

Hybrid quantum-classical neural networks combine conventional deep-learning components with parameterized quantum circuits.

The broader literature explores several approaches, including quantum convolutional neural networks, quantum-classical CNNs, variational quantum classifiers, and parallel quantum-classical architectures.

The architecture in this repository is particularly related to the idea of increasing the **width** of quantum processing through parallel branches rather than relying only on deeper quantum circuits.

Recent work has also investigated parallel quantum-classical convolutional architectures and the relationship between PQC structure, expressibility, entanglement, trainability and robustness.

---

# 📖 References

Useful background:

1. PennyLane — Quantum machine learning framework.
2. PyTorch — Deep learning framework.
3. Research literature on hybrid quantum-classical neural networks.
4. Research literature on parameterized quantum circuits and variational quantum algorithms.
5. Research literature on quantum convolutional neural networks.

For broader context on parallel quantum-classical image-classification architectures, see:

> Liu, H., & Lou, X. *A Parallel Hybrid Quantum-Classical Convolutional Design Using Parameterized Quantum Circuits for Image Classification*, Quantum Engineering, 2026.

The referenced work discusses parallel quantum-classical feature extraction, PQC design, circuit expressibility, entanglement and robustness.

---

# 📝 Citation

If this repository is used in academic work, please cite the corresponding project/paper information associated with the repository.

A BibTeX entry can be added here once the final publication metadata for this specific implementation is available:

```bibtex
@misc{parallel_quantum_classical_image_classification,
  title  = {A Parallel Quantum-Classical Hybrid Architecture for Image Classification},
  author = {Ammar},
  year   = {2026},
  url    = {https://github.com/ammar31300/A-Parallel-Quantum-Classical-Hybrid-Architecture-for-Image-Classification}
}
```

---

# 👨‍💻 Project Structure

A recommended repository structure is:

```text
A-Parallel-Quantum-Classical-Hybrid-Architecture-for-Image-Classification/
│
├── paralell-hybridmoded.ipynb
├── SeekThermal_parallel_HybridModel.ipynb
│
├── README.md
│
├── data/
│   └── ...
│
├── results/
│   ├── models/
│   ├── reports/
│   └── figures/
│
└── requirements.txt
```

---

# ⭐ Key Takeaways

The main concept of this project can be summarized as:

```text
Classical CNN
     +
Parallel Quantum Neural Networks
     +
Multiple Quantum Encodings
     +
Trainable Variational Circuits
     +
Quantum Measurements
     +
Classical Classification
     =
Parallel Hybrid Quantum-Classical Image Classifier
```

The architecture is designed to explore how multiple shallow quantum processing branches can complement classical convolutional feature extraction.

The SeekThermal experiment further demonstrates how the same core architecture can be adapted to a different image domain and a three-class thermal object-classification problem.

---

# 📌 Reproducibility Note

The numerical results shown in the repository should always be interpreted together with:

* dataset version
* preprocessing configuration
* random seed
* number of qubits
* quantum circuit depth
* number of parallel heads
* simulator/backend
* optimizer
* learning rate
* number of training epochs

Changing any of these settings can change the final results.

---

# 📄 License

Add the repository's actual license here if one is defined.

If no license is currently included in the repository, users should treat the source code as **all rights reserved** until an explicit open-source license is added.

---

## 🚀 Summary

This project investigates a practical hybrid quantum-classical strategy for image classification:

**CNNs extract spatial information → parallel quantum circuits transform learned features → quantum measurements form a compact representation → a classical classifier produces the final prediction.**

By implementing multiple quantum encoding strategies and comparing them against a classical baseline, the repository provides a useful experimental framework for studying the role of parallel quantum processing in modern image-classification pipelines.
