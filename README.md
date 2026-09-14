# A Parallel Quantum-Classical Hybrid Architecture for Image Classification

A **Parallel Quantum-Classical Hybrid Neural Network (PQCHNN)** for image classification, combining convolutional feature extraction with multiple parallel variational quantum circuits.

The project investigates whether parallel quantum feature transformations can complement classical convolutional representations across datasets with substantially different visual characteristics, ranging from handwritten grayscale digits to natural RGB images and thermal object imagery.

---

## Abstract

This project presents a parallel quantum-classical hybrid architecture for image classification in which a classical convolutional neural network extracts spatial representations and multiple quantum neural network (QNN) branches independently transform these representations into quantum-derived feature vectors.

Instead of using a single quantum circuit, the proposed architecture employs **multiple parallel quantum heads**, allowing different quantum encoding strategies and spatial feature organizations to process complementary aspects of the classical feature map.

Three quantum feature transformation mechanisms are investigated:

1. **Amplitude Encoding**
2. **Dense Angle Encoding**
3. **ZZ-Feature Encoding**

The resulting quantum representations are concatenated and passed to a classical classifier.

The framework is evaluated across three image-classification settings:

* **MNIST** — grayscale handwritten digits
* **CIFAR-10** — natural RGB images
* **SeekThermal** — thermal object images containing Car, Cat, and Man classes

A fully classical CNN baseline is implemented using the same general feature-extraction philosophy, allowing the contribution of the quantum components to be investigated experimentally.

---

# 1. Research Motivation

Classical convolutional neural networks (CNNs) have demonstrated strong performance in visual recognition tasks. However, quantum machine learning provides an alternative mechanism for transforming learned feature representations through quantum states, parameterized rotations, entanglement, and quantum measurements.

The central hypothesis investigated in this project is:

> **Can multiple parallel quantum feature transformations provide complementary representations of classical CNN features for image classification?**

The proposed framework therefore combines:

$$
\text{Image}
\rightarrow
\text{CNN Feature Extraction}
\rightarrow
\text{Parallel Quantum Processing}
\rightarrow
\text{Feature Fusion}
\rightarrow
\text{Classification}
$$

The architecture is designed to preserve the strengths of classical deep learning in spatial feature extraction while using quantum circuits as trainable nonlinear feature transformations.

---

# 2. Main Contributions

The project provides the following components:

* A reusable **classical convolutional feature extractor**
* A **parallel quantum neural network architecture**
* Multiple quantum data-encoding mechanisms
* Amplitude-based quantum encoding
* Dense angle-based encoding
* ZZ-interaction-based feature encoding
* Multiple parallel quantum heads
* Trainable variational quantum circuits
* Ring and skip-style entanglement patterns
* Quantum expectation-value measurements
* A fully classical baseline
* Evaluation on three substantially different image datasets
* Accuracy, Macro-F1 and Weighted-F1 evaluation
* Confusion matrices and classification reports
* Reproducible training configuration
* Checkpointing and early stopping

---

# 3. Overall Architecture

The complete pipeline can be summarized as:

```text
                         Input Image
                              │
                              ▼
                  ┌──────────────────────┐
                  │   Classical CNN      │
                  │   Feature Extractor  │
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

The key architectural principle is **parallel quantum processing rather than a single sequential quantum branch**.

---

# 4. Mathematical Formulation

Let an input image be represented by

$$
X \in \mathbb{R}^{C\times H\times W}
$$

where \(C\) denotes the number of channels.

The classical convolutional encoder is represented by

$$
F = f_{\theta_c}(X)
$$

where \(F\) is the learned spatial feature representation and \(\theta_c\) represents the trainable classical parameters.

The feature representation is subsequently transformed by \(P\) parallel quantum branches:

$$
Q_i = q_{\theta_{q_i}}(F), \qquad i=1,\ldots,P
$$

where \(q_{\theta_{q_i}}\) represents the \(i\)-th parameterized quantum transformation.

The resulting quantum representations are concatenated:

$$
Q =
Q_1 \Vert Q_2 \Vert \cdots \Vert Q_P
$$

where \(\Vert\) denotes feature concatenation.

The final prediction is then obtained using a classical classifier:

$$
\hat{y} =
g_{\theta_c'}(Q)
$$

The complete model can therefore be expressed as

$$
\boxed{
\hat{y}
=
g_{\theta_c'}
\left(
q_{\theta_{q_1}}(f_{\theta_c}(X))
\Vert
\cdots
\Vert
q_{\theta_{q_P}}(f_{\theta_c}(X))
\right)
}
$$

The parameters of both classical and quantum components are optimized jointly using backpropagation through the differentiable quantum simulation framework.

---

# 5. Classical CNN Feature Extractor

The classical component is responsible for extracting spatial features before quantum processing.

The general structure follows:

```text
Input
 │
 ▼
Conv2D
 │
BatchNorm
 │
ReLU
 │
MaxPooling
 │
Conv2D
 │
BatchNorm
 │
ReLU
 │
MaxPooling
 │
Feature Projection
 │
▼
Quantum Input Representation
```

For the first notebook, the CNN backbone uses lightweight convolutional layers with channel progression such as:

$$
C_{in}
\rightarrow 8
\rightarrow 16
$$

followed by spatial pooling and feature projection.

The SeekThermal implementation uses a deeper feature extractor with approximately:

$$
16 \rightarrow 32 \rightarrow 64 \rightarrow 64
$$

channels.

This design allows the same overall hybrid philosophy to be applied to datasets with different image characteristics.

---

# 6. Parallel Quantum Architecture

A major component of the proposed framework is the use of multiple quantum branches.

The default configuration uses:

$$
P=3
$$

parallel quantum heads.

Each head receives a representation derived from the classical CNN feature map and applies a different quantum feature transformation.

The three principal implementations are:

| Quantum Model            | Encoding Strategy   | Main Purpose                                               |
| ------------------------ | ------------------- | ---------------------------------------------------------- |
| HybridAmplitudeParallel  | Amplitude Encoding  | Encodes normalized feature vectors into quantum amplitudes |
| HybridDenseAngleParallel | Angle Encoding      | Maps learned features directly to rotation angles          |
| HybridZZFeatureParallel  | ZZ Feature Encoding | Combines rotations with data-dependent ZZ interactions     |

---

# 7. Quantum Head 1 — Amplitude Encoding

The amplitude-based branch converts a classical vector into the amplitudes of a quantum state.

For \(n\) qubits, the Hilbert-space dimension is

$$
D=2^n
$$

For example, with six qubits:

$$
2^6=64
$$

Therefore, a 64-dimensional feature vector can be directly mapped to a six-qubit quantum state.

Given

$$
x=(x_0,x_1,\ldots,x_{63})
$$

the normalized quantum state is

$$
|\psi(x)\rangle
=
\sum_{i=0}^{63} x_i |i\rangle
$$

subject to

$$
\sum_i |x_i|^2=1
$$

The implementation uses amplitude normalization and zero-padding when necessary.

---

# 8. Spatial Feature Permutation

Before amplitude encoding, the classical feature map can be reorganized using different spatial permutation strategies.

This is motivated by the observation that the ordering of classical features affects how information is represented in a quantum state.

Possible spatial organization strategies include transformations such as:

* Identity ordering
* Serpentine ordering
* Checkerboard ordering
* Row shifting
* Column shifting
* Reverse ordering
* Diagonal wrapping
* Other structured permutations

The goal is not to change the information content, but to expose different spatial organizations of the same learned feature representation to the quantum circuit.

---

# 9. Quantum Head 2 — Dense Angle Encoding

The dense angle encoding branch maps classical features into quantum rotation angles.

A typical transformation is:

$$
\theta_i = \pi\tanh(x_i)
$$

The bounded nonlinear transformation prevents extremely large rotation angles.

The encoded state can then be generated using parameterized rotations such as:

$$
R_X(\theta_i),\qquad R_Y(\theta_i)
$$

or combinations of trainable rotations.

The general structure is:

```text
CNN Features
     │
     ▼
Dense Projection
     │
     ▼
Angle Transformation
     │
     ▼
RX / RY Encoding
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

---

# 10. Quantum Head 3 — ZZ Feature Encoding

The ZZ-based quantum branch introduces feature-dependent two-qubit interactions.

For neighboring qubits \(i,j\), a ZZ interaction can be represented by

$$
e^{-i\phi_{ij}Z_iZ_j}
$$

where \(\phi_{ij}\) depends on the input features and/or a scaling parameter.

The implementation uses a configurable factor:

$$
\text{zz\_scale}=0.25
$$

A typical decomposition of the interaction is:

```text
CNOT
 │
RZ(φ)
 │
CNOT
```

This allows classical feature information to influence correlations between qubits rather than being represented only by independent single-qubit rotations.

---

# 11. Variational Quantum Circuit

Each quantum branch contains trainable variational parameters.

A parameterized single-qubit rotation can be represented as:

$$
Rot(\alpha,\beta,\gamma)
=
R_Z(\gamma)
R_Y(\beta)
R_Z(\alpha)
$$

The circuit therefore contains both:

1. Data-dependent operations
2. Trainable quantum operations

A simplified layer can be expressed as:

$$
U_l(\mathbf{x},\theta_l)
=
U_{\text{entangle}}
U_{\text{var}}(\theta_l)
U_{\text{encode}}(\mathbf{x})
$$

For \(L\) quantum layers:

$$
U(\mathbf{x},\Theta)
=
U_L
\cdots
U_2
U_1
$$

The default SeekThermal configuration uses:

$$
n_{\text{qubits}}=6
$$

and

$$
n_{\text{layers}}=3
$$

per quantum branch.

---

# 12. Entanglement Strategy

Entanglement is introduced to allow information to propagate between quantum features.

The circuit supports structured connectivity patterns including:

* Ring connectivity
* Pairwise interactions
* Ring + skip-style connections
* Layer-dependent connectivity shifts

A ring structure can be represented as:

$$
q_0-q_1-q_2-\cdots-q_{n-1}-q_0
$$

This provides a compact way of creating correlations among neighboring qubits.

The architecture can additionally modify the interaction pattern between layers, allowing different quantum layers to expose different correlation structures.

---

# 13. Quantum Measurements

After the variational circuit, the model does not directly use the full quantum state.

Instead, measurable expectation values are extracted.

For each qubit, the implementation can measure operators such as

$$
\langle X_i\rangle
=
\langle\psi|X_i|\psi\rangle
$$

and

$$
\langle Z_i\rangle
=
\langle\psi|Z_i|\psi\rangle
$$

For six qubits, measuring both \(X\) and \(Z\) gives:

$$
6\times2=12
$$

features per quantum head.

With three parallel heads:

$$
3\times12=36
$$

quantum features are obtained before the final classifier.

---

# 14. Feature Fusion

The output of each quantum branch is concatenated:

$$
Q =
[Q_1,Q_2,Q_3]
$$

For the default six-qubit configuration:

$$
Q\in\mathbb{R}^{36}
$$

The fused representation is then passed to a classical classifier:

$$
\hat{y}
=
WQ+b
$$

or through a small multilayer classification head depending on the experiment.

This creates the final hybrid architecture:

$$
\boxed{
CNN
\rightarrow
Parallel\ QNNs
\rightarrow
Feature\ Fusion
\rightarrow
Classifier
}
$$

---

# 15. Classical Baseline

To determine whether the quantum components provide useful information, the project includes a fully classical model.

The classical baseline uses the same general CNN feature extraction strategy but does not contain quantum circuits.

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

This baseline is essential because a hybrid model should not be evaluated only by absolute accuracy.

Instead, its performance should be compared with a comparable classical architecture.

---

# 16. Datasets

The framework is evaluated across three image datasets.

## 16.1 MNIST

**MNIST** is used as a relatively simple grayscale benchmark.

| Property            | Value              |
| ------------------- | ------------------ |
| Domain              | Handwritten digits |
| Image type          | Grayscale          |
| Original image size | 28 × 28            |
| Model input         | 32 × 32            |
| Channels            | 1                  |
| Number of classes   | 10                 |
| Classes             | Digits 0–9         |

The notebook uses a 32×32 resized representation with normalization:

$$
\mu=0.5,\qquad\sigma=0.5
$$

The experiment shown in the notebook uses a selected fraction of the training data.

The reported execution configuration contains:

```text
Training samples:   10,800
Validation samples: 1,200
Test samples:      10,000
```

The validation subset is separated from the training portion before optimization.

---

## 16.2 CIFAR-10

CIFAR-10 provides a significantly more challenging natural-image classification problem.

| Property            | Value          |
| ------------------- | -------------- |
| Domain              | Natural images |
| Image type          | RGB            |
| Original image size | 32 × 32        |
| Model input         | 32 × 32        |
| Channels            | 3              |
| Number of classes   | 10             |
| Training set        | 50,000 images  |
| Test set            | 10,000 images  |

The ten categories are:

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

The training pipeline uses augmentation including:

* Random horizontal flip
* Random crop
* Padding
* Tensor conversion
* CIFAR-10 normalization

The normalization parameters used are:

$$
\mu=(0.4914,0.4822,0.4465)
$$

and

$$
\sigma=(0.2023,0.1994,0.2010)
$$

The experiment uses a selected fraction of the training data for the demonstrated configuration.

### Reported CIFAR-10 experiment

The notebook reports:

| Model                   | Test Accuracy |   Macro-F1 | Weighted-F1 |  Test Loss | Best Val. Accuracy |
| ----------------------- | ------------: | ---------: | ----------: | ---------: | -----------------: |
| HybridAmplitudeParallel |        0.7236 |     0.7211 |      0.7211 |     0.7965 |             0.6942 |
| FullClassicalBaseline   |    **0.7327** | **0.7314** |  **0.7314** | **0.7605** |         **0.7004** |

Number of epochs:

$$
15
$$

These values are the results recorded in the notebook and should be interpreted as results for the specific experimental configuration rather than as universal performance claims.

---

# 16.3 SeekThermal

The third dataset extends the evaluation from conventional RGB/grayscale datasets to **thermal imagery**.

Dataset structure:

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

The dataset contains three object categories:

| Class     | Training Images | Test Images |
| --------- | --------------: | ----------: |
| Car       |           1,168 |         356 |
| Cat       |           1,782 |         356 |
| Man       |           1,782 |         356 |
| **Total** |       **4,732** |   **1,068** |

The images are resized to:

$$
128\times96
$$

with three input channels in the implementation.

Normalization is:

$$
\mu=(0.5,0.5,0.5)
$$

$$
\sigma=(0.5,0.5,0.5)
$$

---

# 17. SeekThermal Train/Validation Split

The training set is divided using a stratified split.

The validation ratio is:

$$
r_{val}=0.15
$$

and the split uses:

$$
seed=42
$$

Stratification is performed according to class labels to preserve class proportions.

The independent test set is not used during training.

---

# 18. SeekThermal Label-Mapping Issue

An important dataset-cleaning issue was identified in the SeekThermal notebook.

The original `ImageFolder` mapping was:

```text
Car → 0
Cat → 1
Man → 2
```

However, inspection of the folder contents revealed that the semantic content of the `Car` and `Cat` directories had been swapped.

The notebook therefore defines the remapping:

$$
0\rightarrow1
$$

$$
1\rightarrow0
$$

$$
2\rightarrow2
$$

or equivalently:

```python
remap = {
    0: 1,
    1: 0,
    2: 2
}
```

This is an important reproducibility consideration and should be retained in any future version of the dataset pipeline.

---

# 19. Unified Dataset Comparison

| Dataset     | Image Type            | Channels | Input Size | Classes | Main Challenge                    |
| ----------- | --------------------- | -------: | ---------: | ------: | --------------------------------- |
| MNIST       | Grayscale             |        1 |      32×32 |      10 | Simple low-level patterns         |
| CIFAR-10    | RGB                   |        3 |      32×32 |      10 | Natural-image variability         |
| SeekThermal | Thermal/RGB-formatted |        3 |     128×96 |       3 | Thermal-domain object recognition |

This three-dataset design is useful because the model is tested under increasingly different visual conditions rather than being optimized for a single benchmark.

---

# 20. Experimental Models

The main model families are:

### Model A — Full Classical Baseline

$$
X
\rightarrow CNN
\rightarrow Classifier
$$

No quantum processing.

---

### Model B — Hybrid Amplitude Parallel

$$
X
\rightarrow CNN
\rightarrow
\{QNN_{Amp}^{(1)},QNN_{Amp}^{(2)},QNN_{Amp}^{(3)}\}
\rightarrow Fusion
\rightarrow Classifier
$$

---

### Model C — Hybrid Dense Angle Parallel

$$
X
\rightarrow CNN
\rightarrow
\{QNN_{Angle}^{(1)},QNN_{Angle}^{(2)},QNN_{Angle}^{(3)}\}
\rightarrow Fusion
\rightarrow Classifier
$$

---

### Model D — Hybrid ZZ Feature Parallel

$$
X
\rightarrow CNN
\rightarrow
\{QNN_{ZZ}^{(1)},QNN_{ZZ}^{(2)},QNN_{ZZ}^{(3)}\}
\rightarrow Fusion
\rightarrow Classifier
$$

---

# 21. Experimental Configuration

The SeekThermal implementation uses approximately:

| Parameter         |             Value |
| ----------------- | ----------------: |
| Random Seed       |                42 |
| Image Height      |               128 |
| Image Width       |                96 |
| Classes           |                 3 |
| Batch Size        |                64 |
| Epochs            |                10 |
| Learning Rate     |       \(10^{-3}\) |
| Weight Decay      |       \(10^{-4}\) 
