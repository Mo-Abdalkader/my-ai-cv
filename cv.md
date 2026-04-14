# Mohamed Abdalkader — AI Engineer

## 👨‍💻 Professional Summary

AI Engineer with 1+ year of experience building production-grade ML systems, specializing in **LLMs, RAG Systems, NLP, and Medical AI**. Computer Science graduate from **Zagazig University** (2023), with a strong foundation in software engineering — including system design, algorithms, data structures, and software architecture principles.

Delivered **280% accuracy improvement** via LoRA fine-tuning and multi-agent architectures. Deployed RAG pipelines indexing 20+ medical textbooks with sub-second retrieval. Reduced model VRAM by 59% through 4-bit quantization. Secured **105,000 EGP (~$3,450 USD)** in competitive research grants from ITAC and ASRT. Shipped 8+ production ML projects with Docker and cloud deployment.

Key soft skills developed through hands-on internship and research experience:
- Technical planning and attention to detail
- Leadership and cross-functional collaboration under pressure
- Research documentation and technical writing
- Fast technology adoption

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

**IEEE National Award (2023):** Recognized nationally for renewable energy forecasting framework spanning 21 deep learning architectures (LSTM, GRU, Transformers).

---

## 🚀 Key Projects

### NeuralChat — Multi-Provider LLM Chatbot ⭐ LLM Project
**Stack:** FastAPI · LangChain · Cohere · OpenAI · Groq · Gemini · SSE Streaming · Railway
**Links:** [GitHub](https://github.com/Mo-Abdalkader/NeuralChat) | [Live Demo](https://neural-chat-demo.up.railway.app)

A production-grade full-stack AI chatbot supporting multiple LLM providers with advanced prompting capabilities.

- Supports **Cohere, OpenAI, Groq, and Gemini** with 14+ models switchable from the UI at runtime — no code changes needed
- **5 prompting modes**: Zero-Shot, Few-Shot, Chain-of-Thought, Memory Chain, Structured JSON output
- **SSE streaming** for word-by-word responses with live typing indicator
- **Per-session memory** with configurable depth (1–20 message pairs)
- **6 AI personas**: Assistant, Engineer, Analyst, Writer, Teacher, Data Scientist
- **Real-time cost & latency tracking** per message including token count and USD cost
- Architecture designed for extensibility — adding a new LLM provider touches exactly 2 files
- Deployed on Railway — FastAPI serves both REST API and frontend as a single unified service

---

### Mo-Bot — Personal AI Telegram Bot ⭐ LLM Project
**Stack:** Cloudflare Workers · Groq API · LLaMA 3.3 70B · JavaScript (ESM) · Telegram Bot API
**Links:** [Try the Bot](https://t.me/MoInfoBot)

A production AI assistant that represents Mohamed professionally — answers questions about his background in Arabic, English, and Arabizi (Arabic written in Latin letters).

- **Three-model failover chain** (LLaMA 3.3 70B → LLaMA 3.1 70B → LLaMA 3.1 8B) — each with independent token pools on Groq free tier; seamlessly switches on rate limit with zero downtime
- **Bilingual + Arabizi detection** — detects Arabic Unicode, Egyptian Arabizi patterns, and responds in the same language automatically
- **Pronoun resolution** — understands follow-up questions like "what are his projects?" using conversation history
- **Smart CV extraction** — keyword matching sends only relevant CV sections to the model per question, reducing token waste
- **ctx.waitUntil()** pattern for fire-and-forget async logging that survives after the Cloudflare Worker response is sent
- **Per-user rate limiting** (30 messages / 15 min) to protect free-tier quota
- **Private channel monitoring** — all conversations forwarded to a private Telegram channel for knowledge base improvement
- User-adjustable **response style** (/style brief/detailed/formal/casual) and **language lock** (/language ar/en/auto)
- Running entirely on free infrastructure: Cloudflare Workers + Groq free tier + GitHub raw file

---

### Medical VLM with Multi-Agent Architecture *(Private Research — NDA)* ⭐ LLM + Medical AI Project
**Stack:** Qwen 2.5 7B · PyTorch · LoRA · FAISS · LangChain · Multi-Agent AI

End-to-end medical AI system combining VLM fine-tuning, multi-agent decision fusion, and RAG-grounded report generation.

- Fine-tuned **Qwen 2.5 7B VLM** with LoRA rank-32 on 35,000 retinal scans
- Retinopathy accuracy: **20% → 76% (+280%)** | Edema: **72% → 94% (+30%)**
- Built **multi-agent ensemble** (2 CNNs + RBF-SVM) for decision fusion — ROUGE: 0.17 → 0.44 | BLEU: 0.01 → 0.18 (18x gain)
- **RAG pipeline** indexing 20+ medical textbooks via FAISS (0.3s latency) — grounded VLM outputs to reduce hallucinations
- **4-bit quantization**: VRAM 17GB → 7GB (−59%) with <0.5% accuracy drop
- Built evaluation framework covering factual accuracy and hallucination detection

---

### Brain Tumor Classification System
**Stack:** CNN · DenseNet · TensorFlow · Streamlit · Docker
**Links:** [GitHub](https://github.com/Mo-Abdalkader/Brain-Tumor) | [Live Demo](https://brain-tumor-cls.streamlit.app/)

- CNN achieving **95% accuracy** on 3,000 brain MRI scans across 4 tumor types: Glioma, Meningioma, Pituitary, No Tumor
- Training data expanded **400% via augmentation** (rotation, flipping, zoom, shifting)
- Dockerized Streamlit web app with **<2s inference**; includes detailed educational content per tumor type
- Visitor analytics and feature map visualization built into the UI

---

### Face Recognition & Similarity System
**Stack:** PyTorch · FaceNet · Hybrid Encoder (GoogleNet + ResNet-18) · Streamlit · CustomTkinter · SQL
**Links:** [GitHub](https://github.com/Mo-Abdalkader/Face-Recognition) | [Live Demo](https://face-similarity-recognition.streamlit.app/)

- Hybrid encoder (GoogleNet + ResNet-18) producing **512-dimensional L2-normalized embeddings** trained with Triplet Loss
- **0.92 accuracy at 15 FPS** on CPU hardware using cosine similarity matching
- Desktop GUI with SQLite database managing **1,000+ identity profiles** with metadata (name, department, role)
- Dual interface: Streamlit web app for demos + CustomTkinter desktop app for production use
- Recognition history tracking, batch processing, and export to CSV

---

### NeuralHand — Hand Gesture Control System
**Stack:** MediaPipe · OpenCV · PyAutoGUI · CustomTkinter · Python
**Links:** [GitHub](https://github.com/Mo-Abdalkader/Neural-Hand)

- Real-time hand tracking using **MediaPipe 21-point landmark detection** at 25–30 FPS
- 10+ distinct gestures controlling: mouse movement, left/right click, scroll, window management, volume control
- Gesture-to-action latency of **30–50ms** with 90–95% recognition accuracy
- Mouse smoothing algorithm, configurable control zones, and cooldown system to prevent accidental triggers
- Professional dark-themed UI with floating always-on-top preview window and session analytics

---

### Fast Food Classification System
**Stack:** EfficientNetB0 · TensorFlow · Transfer Learning · Flask · Streamlit · Plotly
**Links:** [Live Demo](https://fast-food-cls.streamlit.app/)

- EfficientNetB0 transfer learning model fine-tuned on 15,000 images across **10 food categories**
- Last 20 layers retrained; early layers frozen to retain low-level ImageNet features
- <1s inference time; Streamlit dashboard includes feature map visualization and interactive Plotly confidence charts
- Dual deployment: lightweight Flask REST API + full-featured Streamlit dashboard with Food Encyclopedia

---

### MNIST Digit Recognition
**Stack:** Custom CNN · TensorFlow · Flask · Streamlit · Plotly
**Links:** [Live Demo](https://mnist-cls.streamlit.app/)

- Custom CNN achieving **~99% accuracy** on MNIST test set with ~150,000 parameters (~400KB model)
- Architecture: Conv2D (32) → MaxPool → Conv2D (64) → MaxPool → Dense (64) → Dropout → Softmax
- Feature map visualization for both convolutional layers (4×8 and 8×8 grids)
- <100ms inference; dual deployment: Flask API + Streamlit multi-page dashboard

---

### Lung & Colon Cancer Detection
**Stack:** DenseNet · Custom CNN · TensorFlow · Flask · Transfer Learning

- Comparative study across 4 model variants: Custom CNN (grayscale/RGB) + DenseNet (grayscale/RGB)
- **Best result: DenseNet RGB — 99% accuracy, 98% precision, 99% recall** on histopathological images
- Trained on the LC25000 dataset (25,000 histopathological images, 5 classes)
- Flask web app for real-time clinical predictions; data augmentation (rotation, flip, zoom) applied throughout

---

### AI-IoT Renewable Energy Prediction *(Graduation Project — IEEE Award)*
**Stack:** LSTM · GRU · Transformers · Azure IoT Hub · Python
**Links:** [GitHub](https://github.com/Mo-Abdalkader/Renewable-Energy-Prediction)

- Python framework benchmarking **21 deep learning architectures** (LSTM, GRU, Transformers) for solar and wind forecasting
- Achieved **<15% MAPE** on solar/wind energy prediction
- IoT sensor integration with real-time data pipeline on **Azure IoT Hub**
- ITAC grant: LSTM-based solar system with 92% accuracy and 48-hour predictive window
- ASRT grant: Transformer-based energy consumption forecasting, reducing prediction error by 23%
- **IEEE National Recognition Award** recipient

---

### Breast Cancer Detection — DEPI Capstone
**Stack:** DenseNet · TensorFlow · Azure ML · Docker · MLflow

- DenseNet transfer learning model achieving **97% accuracy** on 15,000+ histopathological images
- End-to-end MLOps pipeline: training → Docker containerization → Azure ML deployment → MLflow tracking
- Developed as part of the 300+ hour DEPI Microsoft Track government scholarship

---

### IEEE Breast Cancer ML Study
**Stack:** Scikit-learn · Pandas · Python · Jupyter

- Comparative study applying 8 classical ML algorithms (Logistic Regression, SVM, Random Forest, Decision Tree, Neural Network, Naive Bayes, Regression, Clustering) on breast cancer dataset
- Evaluated across accuracy, precision, recall, and F1-score; identified Random Forest and SVM as top performers

---

## 🛠️ Technical Skills

### LLM & NLP ⭐ Primary Focus
- **Models**: Qwen 2.5, LLaMA 3, BERT, GPT, Vision-Language Models (VLMs)
- **Fine-tuning**: LoRA, QLoRA (rank-32), Prefix Tuning
- **RAG**: LangChain, LangGraph, FAISS, Chroma, Pinecone
- **Evaluation**: BLEU, ROUGE, Perplexity, Hallucination Detection
- **Prompting**: Zero-Shot, Few-Shot, Chain-of-Thought, Structured Output, Memory Chain
- **Deployment**: Groq API, Cloudflare Workers, Railway, FastAPI

### Deep Learning & Computer Vision
- **Frameworks**: PyTorch, TensorFlow, Hugging Face Transformers
- **Architectures**: CNNs (DenseNet, EfficientNet, ResNet, VGG), YOLO, MediaPipe, FaceNet
- **Optimization**: 4-bit/8-bit Quantization, Model Pruning, Knowledge Distillation, Triplet Loss

### Classical ML & Data Science
- XGBoost, LightGBM, Random Forest, SVM, Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn, Optuna

### MLOps & Deployment
- Docker, MLflow, FastAPI, Flask, Streamlit
- Azure (Azure ML, Azure IoT Hub), AWS (EC2, S3), Railway
- GitHub Actions, CI/CD Pipelines, Cloudflare Workers

### Programming Languages
- **Python** (Expert), Java, SQL, Bash, JavaScript (ESM)

### Databases & Storage
- FAISS, Chroma, Pinecone (vector databases), SQLite, SQL (relational), Git

---

## 🎓 Education

### B.Sc. Computer Science — Zagazig University, Faculty of Computers and Information
**2019 – 2023 | Cairo, Egypt**

A Computer Science degree from the Faculty of Computers and Information provided Mohamed with a strong software engineering foundation, including:
- System design and software architecture principles
- Algorithms, data structures, and computational thinking
- Database systems, computer networks, and operating systems
- Relevant coursework: Machine Learning, Deep Learning, NLP, Computer Vision

**Graduation Project:** Python framework benchmarking 21 DL models (LSTM, GRU, Transformers) for renewable energy forecasting — achieved <15% MAPE — IEEE National Award recipient — secured two competitive research grants (ITAC + ASRT) totaling 105,000 EGP

---

## 📜 Professional Development & Certifications

- **DEPI Machine Learning Engineer Scholarship** — Microsoft Track | 2024 (300+ hrs, Azure MLOps, production ML)
- **Deep Learning Specialization** — DeepLearning.AI / Stanford University / Andrew Ng | 2024
- **Machine Learning Specialization** — DeepLearning.AI / Stanford University / Andrew Ng | 2022

---

## 🌟 Soft Skills

- ML System Architecture & Design (backed by CS degree in software engineering)
- Technical Planning & Attention to Detail
- Leadership & Team Collaboration (developed through internships and research programs)
- Model Debugging & Performance Optimization
- Prompt Engineering & LLM Evaluation
- Research & Technical Documentation
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
- **95–99% Accuracy** across multiple medical imaging projects
- **8+ Production ML Projects** shipped with Docker and cloud deployment
- **100+ LeetCode** algorithm problems solved
- **21 DL architectures** benchmarked in graduation project framework
