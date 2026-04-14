# Mohamed Abdalkader — AI Engineer

## 👨‍💻 Professional Summary

AI Engineer with 1+ year of experience building production-grade ML systems specializing in LLMs, RAG Systems, NLP, and Medical AI. Delivered a **280% accuracy improvement** via LoRA fine-tuning and multi-agent architectures. Deployed RAG pipelines indexing 20+ medical textbooks with sub-second retrieval. Reduced model VRAM by 59% through 4-bit quantization. Secured **105,000 EGP (~$3,450 USD)** in competitive research grants from ITAC and ASRT. Shipped 8+ production ML projects with Docker and cloud deployment.

Currently open to exciting AI/ML opportunities — remote or on-site.

---

## 📧 Contact Information

- **Email**: Mohameed.Abdalkadeer@gmail.com
- **Phone**: +201023277913
- **WhatsApp**: https://wa.me/201023277913
- **LinkedIn**: https://www.linkedin.com/in/mo-abdalkader/
- **GitHub**: https://github.com/Mo-Abdalkader/
- **LeetCode**: https://leetcode.com/Mo-Abdalkader/
- **Location**: Cairo, Egypt

---

## 💼 Professional Experience

### Freelance AI Engineer
**May 2025 – Feb 2026 | Remote**

- Fine-tuned **Qwen 2.5 7B VLM** with LoRA rank-32 on 35,000 retinal images for diabetic retinopathy diagnosis
- Boosted retinopathy detection accuracy from **20% → 76% (+280%)** and edema detection from 72% → 94%
- Cut diagnosis time by **75%** compared to manual review
- Built **multi-agent pipeline** (2 CNNs + RBF-SVM) raising medical report ROUGE from 0.17 → 0.44 (2.5x gain) and BLEU from 0.01 → 0.18 (18x gain)
- Deployed **RAG system** indexing 20+ medical textbooks via FAISS with 0.3s retrieval latency; served through Flask API in Docker
- Reduced inference memory from **17GB → 7GB (−59%)** via 4-bit quantization with negligible accuracy drop (<0.5%)

### AI Research Intern — Neuronetix
**August 2024 – September 2024**

- Built XGBoost + Random Forest ensemble on 5,000+ patient records; achieved **AUC 0.92, F1 0.89**
- **Ranked Top 3 of 150** in Customer Churn Prediction challenge with **93.5% accuracy** using gradient-boosted trees
- Authored model cards and deployment documentation for production readiness

### Machine Learning Engineer — DEPI (Microsoft Track)
**April 2024 – October 2024**

- 6-month government scholarship program (300+ hours) covering production ML, MLOps, and Azure deployment
- Capstone project: Breast cancer detection model using **DenseNet transfer learning — 97% accuracy** on 15,000+ images
- Skills covered: Azure ML, CI/CD, Docker, MLflow, end-to-end production pipelines

### Machine Learning Intern — ShAI
**March 2024 – June 2024**

- Music genre classifier using MFCC + XGBoost achieving **95% accuracy** across 8 classes
- Diamond price prediction model with **MAE < $50** via feature engineering and Optuna hyperparameter tuning

---

## 🏆 Research & Awards

| Award | Amount | Organization | Year |
|-------|--------|--------------|------|
| Research Grant | 70,000 EGP (~$2,300 USD) | ITAC | 2023 |
| Research Grant | 35,000 EGP (~$1,150 USD) | ASRT | 2023 |
| National Recognition Award | — | IEEE Graduation Project Competition | 2023 |

**ITAC Grant (2023):** Built a cloud-monitored solar generation system with LSTM models achieving 92% accuracy with a 48-hour predictive window.

**ASRT Grant (2023):** Integrated IoT sensors with Transformer-based forecasting for energy consumption estimation; reduced prediction error by 23%.

**IEEE National Award (2023):** Recognized nationally for renewable energy forecasting framework spanning 21 deep learning architectures.

---

## 🚀 Key Projects

### NeuralChat — Multi-Provider LLM Chatbot
**Stack:** FastAPI · LangChain · Cohere · OpenAI · Groq · SSE Streaming · Railway
**Links:** [GitHub](https://github.com/Mo-Abdalkader/NeuralChat) | [Live Demo](https://neuralchat-production.up.railway.app/)

- Multi-provider chatbot supporting **Cohere, OpenAI, and Groq** with 10+ models switchable from the UI
- 5 prompting modes: Zero-Shot, Few-Shot, Chain-of-Thought, Memory Chain, Structured JSON output
- SSE streaming, per-session memory, 6 AI personas, real-time cost & latency tracking
- Deployed on Railway — FastAPI serves both REST API and frontend as a single unified service

### Medical VLM with Multi-Agent Architecture *(Private Research — NDA)*
**Stack:** Qwen 2.5 7B · PyTorch · LoRA · FAISS · Multi-Agent AI

- Fine-tuned Qwen 2.5 7B with LoRA rank-32 on 35,000 retinal scans
- Retinopathy accuracy: **20% → 76% (+280%)** | Edema: **72% → 94% (+30%)**
- Multi-agent ensemble (2 CNNs + RBF-SVM) for decision fusion — ROUGE: 0.17 → 0.44 | BLEU: 0.01 → 0.18
- RAG pipeline indexing 20+ medical textbooks via FAISS (0.3s latency)
- 4-bit quantization: VRAM 17GB → 7GB (−59%)

### Brain Tumor Classification System
**Stack:** CNN · TensorFlow · Streamlit · Docker
**Links:** [GitHub](https://github.com/Mo-Abdalkader/Brain-Tumor) | [Live Demo](https://brain-tumor-cls.streamlit.app/)

- CNN achieving **95% accuracy** on 3,000 brain MRI scans across 4 tumor types
- Dockerized Streamlit web app with <2s inference; training data expanded 400% via augmentation

### Face Recognition & Similarity System
**Stack:** PyTorch · FaceNet · Streamlit · SQL
**Links:** [GitHub](https://github.com/Mo-Abdalkader/Face-Recognition-) | [Live Demo](https://facematch-pro.streamlit.app/)

- Facial recognition with desktop GUI and SQL database managing 1,000+ identity profiles
- Cosine similarity matching — **0.92 accuracy at 15 FPS** on CPU hardware

### AI-IoT Renewable Energy Prediction *(Graduation Project — IEEE Award)*
**Stack:** LSTM · GRU · Transformers · Azure IoT Hub
**Links:** [GitHub](https://github.com/Mo-Abdalkader/Graduation_project)

- 21 deep learning models for solar and wind energy forecasting — **<15% MAPE**
- IoT sensor integration with cloud deployment on Azure IoT Hub
- Recipient of IEEE National Recognition Award

---

## 🛠️ Technical Skills

### LLM & NLP
- **Models**: Qwen 2.5, LLaMA 3, BERT, GPT, Vision-Language Models (VLMs)
- **Fine-tuning**: LoRA, QLoRA (rank-32), Prefix Tuning
- **RAG**: LangChain, LangGraph, FAISS, Chroma, Pinecone
- **Evaluation**: BLEU, ROUGE, Perplexity, Hallucination Detection
- **Prompting**: Zero-Shot, Few-Shot, Chain-of-Thought, Structured Output, Memory Chain

### Deep Learning & Computer Vision
- **Frameworks**: PyTorch, TensorFlow, Hugging Face Transformers
- **Architectures**: CNNs (DenseNet, EfficientNet, ResNet, VGG), YOLO, MediaPipe
- **Optimization**: 4-bit/8-bit Quantization, Model Pruning, Knowledge Distillation

### Classical ML & Data Science
- XGBoost, LightGBM, Random Forest, SVM, Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn, Optuna

### MLOps & Deployment
- Docker, MLflow, FastAPI, Flask, Streamlit
- Azure (Azure ML, Azure IoT Hub), AWS (EC2, S3), Railway
- GitHub Actions, CI/CD Pipelines

### Programming Languages
- **Python** (Expert), Java, SQL, Bash

### Databases & Storage
- FAISS, Chroma, Pinecone (vector databases), SQL (relational), Git

---

## 🎓 Education

### B.Sc. Computer Science — Zagazig University
**2019 – 2023 | Cairo, Egypt**

- Relevant coursework: Machine Learning, Deep Learning, NLP, Computer Vision, Algorithms, Databases
- Graduation Project: Python framework with 21 DL models (LSTM, GRU, Transformers) for renewable energy forecasting — achieved <15% MAPE on solar/wind prediction — IEEE National Award recipient

---

## 📜 Professional Development & Certifications

- **DEPI Machine Learning Engineer Scholarship** — Microsoft Track | 2024 (300+ hrs, Azure MLOps, production ML)
- **Deep Learning Specialization** — DeepLearning.AI / Stanford University / Andrew Ng | 2024
- **Machine Learning Specialization** — DeepLearning.AI / Stanford University / Andrew Ng | 2022

---

## 🌟 Soft Skills

- ML System Architecture & Design
- Model Debugging & Performance Optimization
- Prompt Engineering & LLM Evaluation
- Research & Technical Documentation
- Cross-functional Collaboration
- Algorithm Problem-Solving (100+ LeetCode problems solved)
- Fast Technology Adoption

---

## 🌐 Languages

- **Arabic** — Native
- **English** — Professional Working Proficiency

---

## 🎯 Career Goals

Mohamed is actively seeking AI/ML Engineering roles — remote or on-site — with focus on:
- LLM fine-tuning and RAG system development
- Production ML deployment and MLOps
- Medical AI and computer vision applications
- NLP and conversational AI systems

He is open to full-time positions, freelance contracts, and research collaborations. Best way to reach him is via WhatsApp (https://wa.me/201023277913) or email (Mohameed.Abdalkadeer@gmail.com).

---

## 🏅 Key Achievements at a Glance

- **280% Accuracy Improvement** — diabetic retinopathy (20% → 76%) via LoRA fine-tuning
- **105,000 EGP** in competitive research grants (ITAC + ASRT)
- **Top 3 of 150** — Neuronetix Customer Churn Prediction Challenge (93.5% accuracy)
- **IEEE National Recognition** — Graduation project in renewable energy forecasting
- **2.5x ROUGE Gain** — medical report quality via multi-agent architecture
- **59% VRAM Reduction** — 17GB → 7GB via 4-bit quantization
- **95–97% Accuracy** across multiple medical imaging projects
- **8+ Production ML Projects** shipped with Docker and cloud deployment
- **100+ LeetCode** algorithm problems solved
