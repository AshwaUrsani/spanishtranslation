# 🌐 English to Spanish Neural Machine Translator

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Live%20Demo-Available-brightgreen)

> A fully deployable English-to-Spanish translation system built from scratch using a custom Transformer architecture — trained, saved, and served via a live web app.

🔗 **[Try the Live App →](https://spanishtranslation.streamlit.app)**

---

## 🧠 What Makes This Project Stand Out

Most NLP tutorials stop at training. This project goes further:

- ✅ Custom Transformer built and trained from scratch (no pre-trained models)
- ✅ Solved real serialization challenges to make the model **reloadable and reusable**
- ✅ Deployed as a **production-ready web app** on Streamlit Cloud
- ✅ End-to-end pipeline: raw text → preprocessing → inference → output

---

## 🏗️ Model Architecture

Built with **TensorFlow / Keras** using a sequence-to-sequence Transformer:

| Component | Configuration |
|---|---|
| Transformer Layers | 4 |
| Embedding Dimensions | 128 |
| Attention Heads | 8 |
| Feedforward Dimension | 512 |
| Training Epochs | 10 |
| Validation Accuracy | ~70% |

**Key architectural choices:**
- **Positional Encoding** — injects token order into attention-based layers
- **Multi-Head Self-Attention** — captures cross-lingual semantic relationships
- **Greedy Decoding** — generates the most probable Spanish token at each step

---

## 🔧 Engineering Highlights

### Vectorization Preservation
A common production pitfall: vocabularies that change between training and inference, causing silent errors. This was solved by:
- Using a custom `TextVectorization` layer with `@register_keras_serializable()`
- Saving both source (English) and target (Spanish) vectorizers as `.keras` files
- Ensuring token-index mappings remain **identical** across training and deployment

### Modular Inference Pipeline
- Model weights saved in `.h5` format
- Reconstructed at inference time with a dummy batch pass to rebuild layer shapes
- Enables **repeated use without retraining** — true production behaviour

---

## 🚀 Streamlit Application

**Live app:** [https://spanishtranslation.streamlit.app](https://spanishtranslation.streamlit.app)


---


## 💼 Skills Demonstrated

- Deep Learning & NLP (Transformer architecture from scratch)
- Model serialization and production-safe inference pipelines
- Full-stack ML deployment (Streamlit Cloud + GitHub)
- Software engineering best practices in ML (modularity, reproducibility)

---

*Built as part of CIS433 — AI and Deep Learning | Simon Business School, University of Rochester*
