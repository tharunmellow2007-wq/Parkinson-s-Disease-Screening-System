# 🏥 Parkinson's Disease Detection System

A professional-grade AI-powered voice analysis platform for detecting Parkinson's Disease and classifying disease progression stages using advanced machine learning and signal processing techniques.

## 🌟 Features

- **Disease Detection**: Analyzes 15 voice biomarkers to detect Parkinson's Disease
- **Stage Classification**: Classifies disease progression into 4 stages (Early, Mild, Moderate, Severe)
- **Professional Interface**: Modern, medical-grade UI built with Gradio
- **Advanced Audio Processing**: Includes noise reduction, bandpass filtering, and voice activity detection
- **Comprehensive Analysis**: Provides detailed clinical descriptions, symptoms, and recommendations
- **Visual References**: Displays clinical images for each stage

## 📋 System Overview

This system utilizes:
- **15 Voice Biomarkers**:
  - Jitter Features (4): Frequency variation measures
  - Shimmer Features (6): Amplitude variation measures
  - Harmonicity Features (2): Voice quality indicators (NHR, HNR)
  - Nonlinear Features (3): Complexity measures (RPDE, DFA, PPE)
  
- **Weighted Voting Mechanism**: For accurate stage classification
- **Advanced Signal Processing**: For high-quality feature extraction

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone or download the repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare model files** (if available):
   - Place your trained model files in the `./models` directory:
     - `parkinsons_ensemble_model.pkl`
     - `feature_scaler.pkl`
     - `feature_info.json`

4. **Prepare stage images** (optional):
   - Place stage reference images in the `./images` directory:
     - Format: `stage1a.png`, `stage1b.png`, `stage2a.jpg`, etc.

### Running the Application

```bash
python parkinsons_detection_app.py
```

The application will launch at `http://localhost:7860`

## 🐳 Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY parkinsons_detection_app.py .
COPY models/ ./models/
COPY images/ ./images/

EXPOSE 7860

CMD ["python", "parkinsons_detection_app.py"]
```

Build and run:
```bash
docker build -t parkinsons-detection .
docker run -p 7860:7860 parkinsons-detection
```

## ☁️ Render Deployment

1. **Create a `render.yaml` file**:

```yaml
services:
  - type: web
    name: parkinsons-detection
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python parkinsons_detection_app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
      - key: MODEL_DIR
        value: ./models
      - key: OUTPUT_DIR
        value: ./results
      - key: IMAGES_DIR
        value: ./images
```

2. **Push to GitHub** and connect to Render

3. **Deploy** from Render dashboard

## 🎯 Usage Guide

### Disease Detection Tab

1. **Record or Upload Audio**:
   - Use the microphone to record your voice (5-10 seconds)
   - Or upload an audio file (WAV, MP3, FLAC)

2. **Adjust Settings**:
   - Select noise reduction strength (light/medium/heavy)

3. **Analyze**:
   - Click "🔬 Analyze Audio"
   - View results and extracted features

### Stage Classification Tab

1. **Automatic Analysis**:
   - After disease detection, stage classification runs automatically
   - View detailed stage information, symptoms, and recommendations

2. **Visual References**:
   - Clinical images are displayed for the diagnosed stage

## 🔧 Configuration

Environment variables for deployment:

```bash
MODEL_DIR=./models          # Directory containing model files
OUTPUT_DIR=./results        # Directory for output files
IMAGES_DIR=./images         # Directory containing stage images
```

## 📊 Model Training

If you need to train your own model:

1. Prepare dataset with voice recordings
2. Extract features using the provided functions
3. Train a classification model (e.g., ensemble methods)
4. Save model, scaler, and feature info as pickle/json files

## 🔐 Security Notes

- This is a diagnostic support tool, not a replacement for professional medical diagnosis
- Audio recordings are processed locally and not stored permanently
- Always consult healthcare professionals for medical decisions

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is for educational and research purposes. Consult with legal counsel before using in clinical settings.

## ⚠️ Disclaimer

**IMPORTANT**: This system is designed as a research and educational tool. It should NOT be used as the sole basis for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for proper medical evaluation and care.

## 🆘 Support

For issues or questions:
- Check the documentation
- Review error messages in the interface
- Ensure audio quality meets minimum requirements
- Verify model files are properly loaded

## 📚 References

- Jitter and Shimmer analysis using Praat-Parselmouth
- Feature extraction based on established Parkinson's research
- Weighted voting classification methodology

## 🎨 Interface Features

- **Modern Design**: Professional medical-grade interface
- **Responsive Layout**: Works on desktop and mobile
- **Color-Coded Results**: Easy-to-understand visual feedback
- **Detailed Information**: Comprehensive clinical descriptions
- **Interactive Elements**: Collapsible sections for detailed data

---

**Version**: 1.0  
**Last Updated**: 2025  
**Built with**: Gradio, Librosa, Praat-Parselmouth, scikit-learn
