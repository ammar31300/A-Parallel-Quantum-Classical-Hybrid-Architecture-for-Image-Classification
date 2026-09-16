# A Parallel Quantum-Classical Hybrid Architecture for Image Classification

A **Parallel Quantum-Classical Hybrid Neural Network (PQCHNN)** for image classification that combines classical convolutional feature extraction with multiple parallel variational quantum circuits.

This project investigates whether **parallel quantum feature transformations** can provide complementary representations of classical convolutional features across image datasets with substantially different visual characteristics, including handwritten grayscale digits, natural RGB images, and thermal object imagery.

---

## Abstract

This project presents a parallel quantum-classical hybrid architecture for image classification. A classical convolutional neural network (CNN) first extracts spatial representations from input images. The resulting feature representation is then processed by multiple parallel quantum neural network (QNN) branches, each implementing a different quantum feature transformation.

Unlike a conventional hybrid architecture with a single quantum circuit, the proposed framework employs **multiple parallel quantum heads**. Each head can use a different data-encoding strategy and feature organization, allowing multiple quantum representations of the same classical feature space to be learned and subsequently fused.

Three quantum feature transformation strategies are investigated:

1. **Amplitude Encoding**
2. **Dense Angle Encoding**
3. **ZZ-Feature Encoding**

The outputs of the quantum branches are concatenated and passed to a classical classification head.

The framework is evaluated on three image-classification settings:

* **MNIST** — handwritten grayscale digits
* **CIFAR-10** — natural RGB images
* **SeekThermal** — thermal object imagery with Car, Cat, and Man classes

A fully classical CNN baseline is also implemented using a comparable feature-extraction strategy. This provides a reference point for evaluating the effect of incorporating quantum processing into the architecture.

---

# 1. Research Motivation

Classical convolutional neural networks have demonstrated strong performance in visual recognition tasks. Quantum machine learning, meanwhile, provides alternative mechanisms for transforming learned representations through quantum state preparation, parameterized operations, entanglement, and measurement.

The main research question investigated in this project is:

> **Can multiple parallel quantum feature transformations provide complementary representations of classical CNN features for image classification?**

The proposed framework therefore combines classical spatial feature extraction with parallel quantum transformations:

```text
Image
  │
  ▼
Classical CNN Feature Extraction
  │
  ▼
Parallel Quantum Processing
  │
  ▼
Quantum Feature Fusion
  │
  ▼
Classical Classification Head
  │
  ▼
Prediction
```

The objective is not to replace classical feature extraction, but to investigate whether quantum circuits can serve as trainable feature transformations within an otherwise conventional deep-learning pipeline.

---

# 2. Main Contributions

The project provides the following components:

* A reusable **classical CNN feature extractor**
* A **parallel quantum-classical architecture**
* Multiple quantum data-encoding strategies
* Amplitude-based quantum encoding
* Dense angle-based encoding
* ZZ-interaction-based feature encoding
* Multiple parallel quantum heads
* Trainable variational quantum circuits
* Structured quantum entanglement patterns
* Quantum expectation-value measurements
* A fully classical baseline
* Evaluation across three image datasets
* Accuracy, Macro-F1, and Weighted-F1 metrics
* Confusion matrices and classification reports
* Reproducible training configurations
* Checkpointing and early stopping

---

# 3. Overall Architecture

The complete architecture can be summarized as follows:

```text
                         Input Image
                              │
                              ▼
                  ┌──────────────────────┐
                  │   Classical CNN      │
                  │ Feature Extraction   │
                  └──────────┬───────────┘
                             │
                             ▼
                    Classical Feature Map
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │ Quantum    │ │ Quantum    │ │ Quantum    │
       │ Head 1     │ │ Head 2     │ │ Head 3     │
       │ Amplitude  │ │ Dense      │ │ ZZ Feature │
       │ Encoding   │ │ Angle      │ │ Encoding   │
       └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
             │              │              │
             ▼              ▼              ▼
        Quantum         Quantum        Quantum
        Features        Features       Features
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                    Feature Concatenation
                            │
                            ▼
                  Classical Classification
                            │
                            ▼
                         Prediction
```

The central architectural principle is **parallel quantum processing**, where multiple quantum branches transform classical CNN representations before their outputs are fused.

---

# 4. Mathematical Formulation

Let an input image be represented by

$$
X \in \mathbb{R}^{C \times H \times W},
$$

where \(C\), \(H\), and \(W\) denote the number of channels, height, and width, respectively.

The classical feature extractor is defined as

$$
F = f_{\theta_c}(X),
$$

where \(F\) is the learned feature representation and \(\theta_c\) denotes the trainable classical parameters.

The feature representation is then processed by \(P\) parallel quantum branches:

$$
Q_i = q_{\theta_{q_i}}(F),
\qquad i=1,\ldots,P,
$$

where \(q_{\theta_{q_i}}\) denotes the \(i\)-th parameterized quantum transformation.

The outputs of the quantum branches are concatenated:

$$
Q = Q_1 \Vert Q_2 \Vert \cdots \Vert Q_P,
$$

where \(\Vert\) denotes feature concatenation.

The final prediction is produced by a classical classification head:

$$
\hat{y} = g_{\theta_f}(Q).
$$

The complete hybrid model can therefore be written as

$$
\boxed{
\hat{y}
=
g_{\theta_f}
\left(
q_{\theta_{q_1}}(f_{\theta_c}(X))
\Vert
\cdots
\Vert
q_{\theta_{q_P}}(f_{\theta_c}(X))
\right)
}
$$

The classical and quantum parameters are optimized jointly through the differentiable quantum simulation framework used by the implementation.

---

# 5. Classical CNN Feature Extractor

The classical CNN is responsible for extracting spatial representations before quantum processing.

A representative architecture is:

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

For the MNIST and CIFAR-10 experiments, the CNN backbone uses lightweight convolutional blocks with channel progression similar to:

$$
C_{in} \rightarrow 8 \rightarrow 16.
$$

The SeekThermal configuration uses a deeper feature extractor with approximately:

$$
16 \rightarrow 32 \rightarrow 64 \rightarrow 64.
$$

The exact backbone configuration is dataset-dependent so that the model can accommodate differences in image resolution and visual complexity.

---

# 6. Parallel Quantum Architecture

The proposed framework uses multiple quantum branches to transform the classical feature representation.

The default architecture contains:

$$
P=3
$$

parallel quantum heads.

The three principal quantum transformations are:

| Model                      | Encoding Strategy    | Purpose                                                       |
| -------------------------- | -------------------- | ------------------------------------------------------------- |
| `HybridAmplitudeParallel`  | Amplitude Encoding   | Encodes normalized feature vectors into quantum amplitudes    |
| `HybridDenseAngleParallel` | Dense Angle Encoding | Maps learned features to quantum rotation angles              |
| `HybridZZFeatureParallel`  | ZZ Feature Encoding  | Combines rotations with data-dependent two-qubit interactions |

Depending on the experiment, the three heads may represent different spatial permutations, encoding mechanisms, or combinations of both.

---

# 7. Quantum Head 1 — Amplitude Encoding

The amplitude-based branch maps a classical feature vector to the amplitudes of a quantum state.

For \(n\) qubits, the Hilbert-space dimension is

$$
D=2^n.
$$

For example, with six qubits:

$$
2^6=64.
$$

Therefore, a 64-dimensional feature vector can be represented directly in a six-qubit state.

Given

$$
x=(x_0,x_1,\ldots,x_{63}),
$$

the normalized quantum state is

$$
|\psi(x)\rangle
=
\sum_{i=0}^{63}x_i|i\rangle,
$$

subject to

$$
\sum_i |x_i|^2=1.
$$

The implementation performs normalization and zero-padding when the input dimensionality does not exactly match the required amplitude dimension.

---

# 8. Spatial Feature Permutations

Before amplitude encoding, the classical feature representation can be reorganized using structured spatial permutations.

The motivation is that the ordering of classical features determines how information is distributed across the amplitudes of the quantum state.

The implementation can support transformations such as:

* Identity ordering
* Serpentine ordering
* Checkerboard ordering
* Row shifting
* Column shifting
* Reverse ordering
* Diagonal wrapping
* Other structured permutations

These operations do not change the underlying feature values. Instead, they change their ordering before quantum encoding.

This provides a mechanism for investigating whether different spatial organizations of the same classical representation lead to different quantum feature spaces.

---

# 9. Quantum Head 2 — Dense Angle Encoding

The dense angle encoding branch maps classical features to quantum rotation angles.

A representative transformation is

$$
\theta_i = \pi\tanh(x_i).
$$

The hyperbolic tangent bounds the input before it is converted into a rotation angle.

The encoded representation can then be applied through parameterized rotations such as

$$
R_X(\theta_i)
\quad\text{and}\quad
R_Y(\theta_i),
$$

possibly followed by trainable variational rotations.

A typical processing sequence is:

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

For two qubits \(i\) and \(j\), a ZZ interaction can be represented as

$$
e^{-i\phi_{ij}Z_iZ_j},
$$

where \(\phi_{ij}\) is determined by the encoded features and the selected interaction scale.

The implementation uses a configurable parameter such as

$$
\text{zz\_scale}=0.25.
$$

A common circuit decomposition of the interaction is:

```text
CNOT
  │
RZ(φ)
  │
CNOT
```

This allows the input features to influence correlations between qubits rather than being represented exclusively through independent single-qubit rotations.

---

# 11. Variational Quantum Circuit

Each quantum branch contains trainable variational parameters.

A general single-qubit rotation can be represented as

$$
Rot(\alpha,\beta,\gamma)
=
R_Z(\gamma)
R_Y(\beta)
R_Z(\alpha).
$$

The quantum circuits therefore combine two types of operations:

1. **Data-dependent encoding operations**
2. **Trainable variational operations**

A single quantum layer can be represented as

$$
U_l(\mathbf{x},\theta_l)
=
U_{\mathrm{ent}}
U_{\mathrm{var}}(\theta_l)
U_{\mathrm{enc}}(\mathbf{x}).
$$

For \(L\) layers:

$$
U(\mathbf{x},\Theta)
=
U_L\cdots U_2U_1.
$$

The default SeekThermal configuration uses approximately:

$$
n_{\mathrm{qubits}}=6
$$

and

$$
n_{\mathrm{layers}}=3
$$

per quantum branch.

---

# 12. Entanglement Strategy

Entanglement enables interactions between quantum features and allows information to propagate across qubits.

The implementation supports structured connectivity patterns including:

* Ring connectivity
* Pairwise interactions
* Ring plus skip-style connections
* Layer-dependent connectivity patterns

A ring topology can be represented as

$$
q_0-q_1-q_2-\cdots-q_{n-1}-q_0.
$$

The connectivity pattern can also be modified between layers, allowing different layers to model different interaction structures.

---

# 13. Quantum Measurements

The model does not directly pass the complete quantum state to the classifier. Instead, measurable expectation values are extracted from the final quantum state.

For example, the expectation value of a Pauli-\(X\) operator on qubit \(i\) is

$$
\langle X_i\rangle
=
\langle\psi|X_i|\psi\rangle,
$$

while the corresponding Pauli-\(Z\) expectation value is

$$
\langle Z_i\rangle
=
\langle\psi|Z_i|\psi\rangle.
$$

With six qubits, measuring both \(X\) and \(Z\) observables produces

$$
6\times2=12
$$

features per quantum head.

With three parallel heads, the fused quantum representation therefore contains

$$
3\times12=36
$$

features before the final classifier.

---

# 14. Quantum Feature Fusion

The output of each quantum branch is concatenated:

$$
Q=[Q_1,Q_2,Q_3].
$$

For the default six-qubit configuration:

$$
Q\in\mathbb{R}^{36}.
$$

The fused representation is then passed to a classical classification head:

$$
\hat{y}=WQ+b,
$$

or, depending on the experiment, to a small multilayer classification head.

The resulting hybrid pipeline is:

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

A fully classical baseline is included to provide a reference for evaluating the effect of quantum processing.

The baseline follows the same general feature-extraction philosophy but replaces the quantum branches with a classical classification component:

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

The purpose of this baseline is to determine whether the additional quantum transformations provide measurable differences relative to a comparable classical architecture.

---

# 16. Datasets

The framework is evaluated on three image datasets with different visual characteristics.

## 16.1 MNIST

**MNIST** is used as a relatively simple grayscale image-classification benchmark.

| Property            | Value              |
| ------------------- | ------------------ |
| Domain              | Handwritten digits |
| Image type          | Grayscale          |
| Original image size | 28 × 28            |
| Model input size    | 32 × 32            |
| Channels            | 1                  |
| Number of classes   | 10                 |
| Classes             | Digits 0–9         |

The notebook resizes the images to \(32\times32\) and applies normalization with

$$
\mu=0.5,\qquad\sigma=0.5.
$$

The demonstrated experiment uses a selected subset of the available training data.

### Reported Execution Configuration

| Split      | Samples |
| ---------- | ------: |
| Training   |  10,800 |
| Validation |   1,200 |
| Test       |  10,000 |

The validation subset is separated from the training data before model optimization.

---

## 16.2 CIFAR-10

CIFAR-10 provides a more challenging natural-image classification problem.

| Property            | Value          |
| ------------------- | -------------- |
| Domain              | Natural images |
| Image type          | RGB            |
| Original image size | 32 × 32        |
| Model input size    | 32 × 32        |
| Channels            | 3              |
| Number of classes   | 10             |
| Training set        | 50,000         |
| Test set            | 10,000         |

The ten classes are:

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

The training pipeline includes:

* Random horizontal flip
* Random crop
* Padding
* Tensor conversion
* CIFAR-10 normalization

The normalization parameters are

$$
\mu=(0.4914,0.4822,0.4465)
$$

and

$$
\sigma=(0.2023,0.1994,0.2010).
$$

The demonstrated experiment uses a selected fraction of the available training data.

### Reported CIFAR-10 Experiment

The following results are reported by the notebook for the specified experimental configuration:

| Model                     | Test Accuracy | Macro-F1 | Weighted-F1 | Test Loss | Best Validation Accuracy |
| ------------------------- | ------------: | -------: | ----------: | --------: | -----------------------: |
| `HybridAmplitudeParallel` |        0.7236 |   0.7211 |      0.7211 |    0.7965 |                   0.6942 |
| `FullClassicalBaseline`   |        0.7327 |   0.7314 |      0.7314 |    0.7605 |                   0.7004 |

Number of epochs:

$$
15
$$

These values should be interpreted as results for the reported experimental configuration rather than as general performance claims.

---

## 16.3 SeekThermal

The third dataset extends the evaluation to **thermal object imagery**.

The dataset is organized approximately as follows:

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

The implementation resizes images to

$$
128\times96
$$

and uses three input channels.

Normalization is performed using

$$
\mu=(0.5,0.5,0.5)
$$

and

$$
\sigma=(0.5,0.5,0.5).
$$

---

# 17. SeekThermal Train/Validation Split

The training set is divided into training and validation subsets using a stratified split.

The validation ratio is

$$
r_{\mathrm{val}}=0.15,
$$

with

$$
\mathrm{seed}=42.
$$

Stratification is performed using the class labels to preserve approximately the same class distribution across the two subsets.

The independent test set is kept separate from the training and validation process.

---

# 18. SeekThermal Label-Mapping Issue

An important dataset-preprocessing issue was identified in the SeekThermal notebook.

The original `ImageFolder` mapping was:

```text
Car → 0
Cat → 1
Man → 2
```

However, inspection of the dataset contents indicated that the semantic contents of the `Car` and `Cat` directories had been exchanged.

The notebook therefore applies the following remapping:

$$
0\rightarrow1
$$

$$
1\rightarrow0
$$

$$
2\rightarrow2
$$

implemented as:

```python
remap = {
    0: 1,
    1: 0,
    2: 2,
}
```

This preprocessing step is important for reproducibility and should be documented whenever the same dataset version is used.

---

# 19. Dataset Comparison

| Dataset     | Image Type      | Channels | Input Size | Classes | Primary Challenge               |
| ----------- | --------------- | -------: | ---------: | ------: | ------------------------------- |
| MNIST       | Grayscale       |        1 |    32 × 32 |      10 | Simple visual patterns          |
| CIFAR-10    | RGB             |        3 |    32 × 32 |      10 | Natural-image variability       |
| SeekThermal | Thermal imagery |        3 |   128 × 96 |       3 | Cross-domain object recognition |

Using three visually different datasets allows the architecture to be evaluated beyond a single benchmark and provides a broader view of its behavior under different input characteristics.

---

# 20. Experimental Models

The main model families are defined as follows.

## Model A — Full Classical Baseline

$$
X
\rightarrow CNN
\rightarrow Classifier
$$

No quantum processing is applied.

---

## Model B — Hybrid Amplitude Parallel

$$
X
\rightarrow CNN
\rightarrow
\{QNN_{\mathrm{Amp}}^{(1)},
QNN_{\mathrm{Amp}}^{(2)},
QNN_{\mathrm{Amp}}^{(3)}\}
\rightarrow Fusion
\rightarrow Classifier
$$

---

## Model C — Hybrid Dense Angle Parallel

$$
X
\rightarrow CNN
\rightarrow
\{QNN_{\mathrm{Angle}}^{(1)},
QNN_{\mathrm{Angle}}^{(2)},
QNN_{\mathrm{Angle}}^{(3)}\}
\rightarrow Fusion
\rightarrow Classifier
$$

---

## Model D — Hybrid ZZ Feature Parallel

$$
X
\rightarrow CNN
\rightarrow
\{QNN_{\mathrm{ZZ}}^{(1)},
QNN_{\mathrm{ZZ}}^{(2)},
QNN_{\mathrm{ZZ}}^{(3)}\}
\rightarrow Fusion
\rightarrow Classifier
$$

---

# 21. Experimental Configuration

A representative SeekThermal configuration is:

| Parameter               |       Value |
| ----------------------- | ----------: |
| Random Seed             |          42 |
| Image Height            |         128 |
| Image Width             |          96 |
| Number of Classes       |           3 |
| Batch Size              |          64 |
| Number of Epochs        |          10 |
| Learning Rate           | \(10^{-3}\) |
| Weight Decay            | \(10^{-4}\) |
| Number of Qubits        |           6 |
| Quantum Layers          |           3 |
| Number of Quantum Heads |           3 |
| ZZ Scale                |        0.25 |
| Validation Ratio        |        0.15 |

> **Note:** The exact configuration may vary between datasets and experiments. The values above describe the representative SeekThermal configuration and should not be assumed to apply to every experiment in the project.

---

# 22. Training Procedure

The hybrid models are trained end-to-end.

The general training pipeline is:

```text
Input Batch
    │
    ▼
CNN Forward Pass
    │
    ▼
Feature Representation
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
 Quantum Head 1  Quantum Head 2  Quantum Head 3
    │              │              │
    └──────────────┼──────────────┘
                   ▼
            Feature Concatenation
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

The training configuration includes:

* Mini-batch optimization
* Validation after training epochs
* Checkpointing of the best validation model
* Early stopping when configured
* Reproducible random seeds
* Final evaluation on an independent test set

---

# 23. Evaluation Metrics

Model performance is evaluated using multiple classification metrics.

### Accuracy

$$
\mathrm{Accuracy}
=
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
$$

### Macro-F1

Macro-F1 calculates the F1 score independently for each class and then averages the results:

$$
F1_{\mathrm{macro}}
=
\frac{1}{K}
\sum_{k=1}^{K}F1_k.
$$

This metric gives equal importance to each class.

### Weighted-F1

Weighted-F1 calculates the class-wise F1 scores and weights them according to class support:

$$
F1_{\mathrm{weighted}}
=
\sum_{k=1}^{K}
w_kF1_k.
$$

This is particularly useful when class frequencies are not perfectly balanced.

### Additional Evaluation

The project also generates:

* Confusion matrices
* Per-class precision
* Per-class recall
* Per-class F1-score
* Classification reports
* Training and validation loss curves
* Training and validation accuracy curves

---

# 24. Reproducibility

Experiments use explicit random seeds where applicable.

The primary seed used in the reported configurations is:

```python
SEED = 42
```

For reproducibility, the following should be recorded for each experiment:

* Dataset version
* Dataset split
* Random seed
* Image preprocessing
* Data augmentation
* CNN architecture
* Number of qubits
* Number of quantum layers
* Quantum encoding strategy
* Entanglement pattern
* Optimizer
* Learning rate
* Weight decay
* Batch size
* Number of epochs
* Early-stopping configuration
* Hardware and software environment

Because quantum simulation can be computationally expensive, recording the exact configuration is especially important when comparing experiments.

---

# 25. Project Structure

A recommended project structure is:

```text
project/
├── README.md
├── requirements.txt
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
├── checkpoints/
├── results/
│   ├── metrics/
│   ├── figures/
│   └── confusion_matrices/
│
└── configs/
    ├── mnist.yaml
    ├── cifar10.yaml
    └── seekthermal.yaml
```

The exact structure can be adapted to the implementation.

---

# 26. Computational Considerations

Quantum simulation introduces additional computational overhead compared with a purely classical CNN.

In particular, computational cost is affected by:

* Number of qubits
* Number of quantum layers
* Number of parallel quantum heads
* Number of expectation values
* Batch size
* Simulation backend
* Differentiation method

Increasing the number of quantum branches therefore increases both the representational capacity and the computational cost of the model.

For this reason, the project focuses on relatively small quantum circuits that can be simulated on classical hardware.

---

# 27. Limitations

Several limitations should be considered when interpreting the experimental results.

### 27.1 Quantum Simulation

The experiments are performed using quantum simulation rather than execution on large-scale fault-tolerant quantum hardware.

Therefore, simulation results should not automatically be interpreted as evidence of practical quantum advantage.

### 27.2 Small Quantum Circuits

The number of qubits and circuit depth are intentionally limited because of simulation cost.

Larger circuits may provide additional representational capacity but can also introduce substantially higher computational requirements.

### 27.3 Dataset-Specific Configurations

The CNN backbone, preprocessing pipeline, training subset, and quantum configuration can vary between datasets.

Consequently, comparisons across datasets should be interpreted as cross-domain experiments rather than as a controlled benchmark with identical hyperparameters.

### 27.4 Limited Ablation Coverage

To isolate the contribution of individual architectural components, additional ablation experiments would be useful, including:

* Single quantum head vs. multiple heads
* Different numbers of quantum heads
* Different numbers of qubits
* Different circuit depths
* Different entanglement patterns
* Different feature permutations
* Quantum vs. classical feature projections

---

# 28. Future Work

Potential extensions include:

1. **Systematic ablation studies**
2. **More quantum encoding strategies**
3. **Alternative entanglement topologies**
4. **Different quantum measurement schemes**
5. **Noise-aware quantum simulation**
6. **Execution on real quantum hardware**
7. **Larger and more diverse thermal datasets**
8. **Parameter-count and computational-cost comparisons**
9. **Statistical significance testing across repeated runs**
10. **Comparison with stronger classical CNN baselines**
11. **Automated search over quantum circuit architectures**
12. **Investigation of learned quantum representations using dimensionality-reduction techniques**

A particularly important direction is to evaluate whether improvements, if observed, remain consistent across multiple random seeds and controlled ablation settings.

---

# 29. Research Questions

The project is designed to investigate several questions:

### RQ1 — Parallel Quantum Processing

Does using multiple quantum branches provide different information compared with a single quantum branch?

### RQ2 — Encoding Strategy

How do amplitude, angle, and ZZ-based feature encodings affect classification performance?

### RQ3 — Feature Organization

Does changing the ordering of classical spatial features influence the resulting quantum representation?

### RQ4 — Dataset Dependence

Does the usefulness of quantum feature transformations change across grayscale, natural RGB, and thermal imagery?

### RQ5 — Classical Comparison

How does the hybrid architecture compare with a comparable classical feature-extraction and classification pipeline?

### RQ6 — Computational Cost

What is the relationship between quantum circuit complexity and classification performance?

---

# 30. Summary

This project implements a **parallel quantum-classical architecture for image classification** in which classical CNN features are transformed by multiple quantum neural network branches.

The framework combines:

$$
\boxed{
\text{CNN Feature Extraction}
+
\text{Parallel Quantum Transformations}
+
\text{Feature Fusion}
+
\text{Classical Classification}
}
$$

Three main quantum encoding strategies are investigated:

* **Amplitude Encoding**
* **Dense Angle Encoding**
* **ZZ Feature Encoding**

The architecture is evaluated on:

* **MNIST**
* **CIFAR-10**
* **SeekThermal**

The inclusion of a fully classical baseline enables direct experimental comparison between classical and hybrid representations.

Overall, the project provides a framework for investigating **parallel quantum feature transformations within convolutional image-classification pipelines**, while emphasizing reproducibility, controlled experimentation, and evaluation across heterogeneous image domains.

---

## Status

This repository represents an experimental research implementation.

Results should be interpreted in the context of the specific datasets, preprocessing pipelines, model configurations, simulation backend, and training procedures used in each experiment.

For scientifically meaningful comparisons, experiments should ideally be repeated across multiple random seeds and accompanied by controlled ablation studies.

---

## License

Add the project license here, for example:

```text
MIT License
```

if the repository is intended to be released under the MIT License.

---

## Citation

If this project is used in academic work, add the appropriate citation information here.

```bibtex
@software{pqchnn,
  title  = {A Parallel Quantum-Classical Hybrid Architecture for Image Classification},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/your-username/your-repository}
}
```
