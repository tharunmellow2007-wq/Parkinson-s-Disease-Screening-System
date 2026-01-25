#NeuroVox: Speech-Based Parkinson’s Disease Screening System

📌 Overview

NeuroVox is an AI-based, non-contact screening system designed to support the early detection of Parkinson’s Disease (PD) using speech analysis. Since vocal impairments are among the earliest manifestations of PD, this project leverages acoustic biomarkers to enable accessible, low-cost, and scalable screening before severe disease progression.

🎯 Problem Statement

Parkinson’s Disease is commonly diagnosed only in mid or late stages, when irreversible neurological damage has already occurred. Current diagnostic methods rely on subjective clinical assessment and expensive neuroimaging scans, limiting early intervention—especially in rural and underserved regions.

💡 Proposed Solution

NeuroVox analyzes subtle changes in speech using acoustic feature extraction and machine learning models to identify Parkinson’s-related patterns at an early stage. The system functions as a screening and decision-support tool, not a replacement for medical diagnosis, and aims to assist clinicians with timely insights.

🧠 Technical Approach

Voice data acquisition using standard microphones

Noise reduction and speech preprocessing

Extraction of clinically relevant acoustic features

Dual-stage AI architecture:

PD vs Healthy classification

Disease stage estimation

Instant result generation via a digital interface or API

🎙️ Acoustic Features Used

Fundamental frequency (F0)

Jitter & Shimmer

Harmonics-to-Noise Ratio (HNR)

Noise-to-Harmonics Ratio (NHR)

MFCCs

RPDE, DFA, PPE

RAP, PPQ

These features capture vocal instability and motor speech impairments associated with Parkinson’s Disease.

🚀 Key Features

Non-invasive and contactless screening

Speech-based early risk assessment

Low-cost and hardware-independent

AI/ML-based classification models

API-ready architecture for deployment

Designed for research and clinical support

🧰 Technology Stack

Language: Python

Audio Processing: Librosa, Praat, NumPy, SciPy

Machine Learning: SVM, Random Forest, XGBoost, KNN, Neural Networks

Development & Training: Google Colab (Jupyter Notebook)

Deployment: Falcon / Flask / Streamlit (configurable)

Version Control: Git & GitHub

📂 Project Structure
NeuroVox/
│
├── data/            # Speech datasets & augmented features
├── notebooks/       # Google Colab / Jupyter notebooks
├── src/             # Preprocessing, feature extraction, models
├── models/          # Trained ML models
├── results/         # Evaluation metrics & outputs
├── docs/            # Project documentation
├── requirements.txt # Dependencies
└── README.md

📊 Impact & Benefits

Enables early Parkinson’s screening and timely intervention

Improves access to neurological screening in remote and underserved areas

Reduces dependency on costly scans and specialist visits

Supports SDGs: 3 (Good Health), 9 (Innovation), 10 (Reduced Inequality)

🔮 Future Scope

Multilingual and noise-robust speech models

Integration with mobile and telemedicine platforms

Multimodal fusion (speech + gait + handwriting)

Large-scale clinical validation

Edge deployment for real-time screening

⚠️ Disclaimer

This project is intended for research and educational purposes only.
It is not a medical diagnostic tool and should not replace professional clinical evaluation.

📜 License

This project is licensed under the MIT License.
