# 🧠 A Sentiment-Aware Transformer Framework for EEG-Based Emotion and Cognitive Load Analysis

This repository implements a novel **Sentiment-Aware Transformer Network (SATN)** designed to analyze emotional states and cognitive load using EEG signals. It introduces an advanced modeling framework that fuses deep learning with domain-specific sentiment information, enabling high-resolution, real-time tracking of mental states — particularly applicable in competitive and high-pressure environments such as sports.

---

## 📌 Key Features

- 🎯 **EEG-Based Emotion & Cognitive Load Detection**
- 🧠 **Transformer-based Architecture** for sequential EEG signal modeling
- 💬 **Sentiment Lexicon Integration** into feature learning
- 🔁 **Emotion-Driven Optimization (EDO)** with dynamic weighting
- 📈 **Interpretability and Evaluation** via custom metrics and analysis tools

---

## 📁 Project Structure

```
EEG-Sentiment-Aware-Transformer/
├── config/                  # YAML-based configuration management
├── data/                    # Raw EEG and processed features
├── scripts/                 # Preprocessing, training, evaluation, inference
├── models/                  # SATN and EDO implementation
├── utils/                   # Dataset loader, metrics, logger
├── experiments/             # For ablation and baseline comparisons
├── requirements.txt         # Python dependencies
├── config.yaml              # Main configuration file
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/eeg-sentiment-transformer.git
cd eeg-sentiment-transformer
```

2. **Create environment and install dependencies**

```bash
pip install -r requirements.txt
```

> Optional: Use `conda` or `virtualenv` if preferred.

---

## 📥 Dataset

This framework assumes preprocessed EEG data saved in `.npy` format.

You can preprocess public EEG datasets (e.g., DEAP, SEED) using the script:

```bash
python scripts/preprocess.py
```

- Raw `.edf` EEG files should be stored in `data/raw/`
- Preprocessed features will be saved to `data/processed/`

---

## 🏗️ Model Overview

- **Input**: Extracted EEG feature vectors (e.g., statistical signals per channel)
- **Transformer Encoder**: Captures temporal and spatial dependencies
- **Sentiment Embedding**: Each token is weighted based on domain-specific sentiment scores
- **EDO**: Emotion-Driven Optimization applies weighted losses using sentiment signals
- **Output**: Emotion or cognitive load classification label

---

## 🚀 Quickstart

### Train the model

```bash
python scripts/train.py
```

Trained models will be saved in `checkpoints/`.

### Evaluate the model

```bash
python scripts/evaluate.py
```

### Inference on new data

```bash
python scripts/inference.py
```

Ensure `sample.npy` is a processed EEG feature file.

---

## 🔧 Configuration

You can adjust model/training settings via `config/config.yaml`.

Example:

```yaml
training:
  epochs: 20
  learning_rate: 0.0005
  use_edo: true
model:
  num_heads: 8
  hidden_dim: 512
```

---

## 📊 Metrics & Evaluation

We report:

- Accuracy
- Precision, Recall, F1-score
- Per-class evaluation (via `classification_report`)
- Support for ablation and baseline comparison under `experiments/`

---

## 🧪 Example Experiments

- `experiments/baseline_comparison/`: Compare SATN with CNN/LSTM
- `experiments/ablation_study/`: Disable EDO or lexicon embeddings

---

## 🔮 Future Development

We envision several meaningful extensions and enhancements to this framework to improve its performance, generalization, and real-world applicability:

### 1. Real-time EEG Stream Processing
- Integration with real-time EEG acquisition systems (e.g., OpenBCI, Emotiv) to support live emotion and cognitive monitoring in sports, e-learning, or VR/AR applications.
- Develop asynchronous input pipelines and latency-aware model adaptations.

### 2. Cross-Domain Transfer Learning
- Implement domain adaptation techniques to transfer learned EEG emotion representations from general datasets (e.g., DEAP, SEED) to domain-specific data such as sports, gaming, or military simulations.
- Evaluate few-shot learning and unsupervised pretraining to reduce data dependence.

### 3. Multimodal Fusion
- Combine EEG signals with other physiological or behavioral modalities (e.g., facial expressions, speech, heart rate variability, eye-tracking) for richer affective computing.
- Design transformer-based cross-modal encoders or fusion heads.

### 4. Neurofeedback & Closed-Loop Interfaces
- Incorporate neurofeedback mechanisms for mental resilience training (e.g., alert athletes to cognitive overload or frustration).
- Develop reward-based adaptive feedback using reinforcement learning.

### 5. Explainability and Interpretability
- Use attention visualization, SHAP or LIME to better understand the neural correlates of emotion.
- Design interpretable modules for coaches or sports psychologists to explore mental patterns.

### 6. Deployable Toolkits & APIs
- Wrap the framework into a RESTful API or lightweight mobile/edge application using ONNX or TensorFlow Lite.
- Create customizable dashboards for teams, coaches, or researchers to monitor player states.

---

## 📜 License

This repository is licensed under the **MIT License**.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, provided that:

- You include the original copyright notice.
- You include this license text in all copies or substantial portions of the Software.
- You **do not hold the authors liable** for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise.

**Note**: This license permits **both academic and commercial use**. If you plan to use this work in a publication, please consider citing it or contacting the author for collaboration.

---

## 🙋‍♀️ Acknowledgements

This work builds upon the efforts and contributions of several open-source and academic resources:

### Datasets
- The [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) for emotion analysis using EEG, peripheral physiological signals, and multimedia.
- The [SEED dataset](https://bcmi.sjtu.edu.cn/~seed/index.html) for semantic EEG emotion recognition.

### Open-source Tools
- [PyTorch](https://pytorch.org/) for flexible deep learning development.
- [MNE-Python](https://mne.tools/stable/index.html) for EEG signal preprocessing and visualization.
- [Gensim](https://radimrehurek.com/gensim/) for pretrained word vector models.
- [scikit-learn](https://scikit-learn.org/) for machine learning evaluation and metrics.
- [SentiWordNet](https://sentiwordnet.isti.cnr.it/) and [GloVe](https://nlp.stanford.edu/projects/glove/) for sentiment-aware embedding references.

### Academic Inspiration
- Research in affective computing, particularly "EEG-based emotion recognition using deep learning networks" and "Transformer-based models for sequential physiological signal processing".
- The interdisciplinary guidance of sports psychology, neuroscience, and machine learning communities.

Special thanks to all contributors, collaborators, and open research supporters who made this work possible. If you'd like to collaborate, contribute, or provide feedback, feel free to reach out!

---
