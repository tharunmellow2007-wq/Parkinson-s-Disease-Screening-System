# Parkinson's Disease Detection System - Gradio App

## Overview
A professional medical-grade web application for Parkinson's Disease detection using voice analysis with 15 voice biomarkers and AI-powered stage classification.

## Features
- **Disease Detection**: Analyzes 15 voice biomarkers to detect Parkinson's Disease
- **Stage Classification**: Classifies disease progression into 4 stages (Early, Mild, Moderate, Severe)
- **Audio Processing**: Advanced noise reduction and signal processing
- **Professional UI**: Modern, medical-grade interface with comprehensive visualizations
- **Dual Input**: Support for both recording and file upload
- **Clinical Insights**: Detailed symptoms, characteristics, and medical recommendations

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Model Files
Create the following directory structure:
```
./models/
  ├── parkinsons_ensemble_model.pkl
  ├── feature_scaler.pkl
  └── feature_info.json

./images/
  ├── stage1a.png
  ├── stage1b.png
  ├── stage2a.png
  ├── stage2b.png
  ├── stage3a.png
  ├── stage3b.png
  ├── stage4a.png
  └── stage4b.png

./outputs/  # Created automatically
```

### 3. Configure Paths (Optional)
Set environment variables for custom paths:
```bash
export MODEL_DIR=/path/to/models
export OUTPUT_DIR=/path/to/outputs
export IMAGES_DIR=/path/to/images
```

## Running the Application

### Local Development
```bash
python parkinsons_gradio_app.py
```

The app will be available at: `http://localhost:7860`

### Production Deployment (Render)

#### Option 1: Using Render Web Service

1. **Create a `render.yaml` file:**
```yaml
services:
  - type: web
    name: parkinsons-detection
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python parkinsons_gradio_app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: MODEL_DIR
        value: ./models
      - key: OUTPUT_DIR
        value: ./outputs
      - key: IMAGES_DIR
        value: ./images
```

2. **Push to GitHub**
3. **Connect to Render**
4. **Deploy**

#### Option 2: Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "parkinsons_gradio_app.py"]
```

Build and run:
```bash
docker build -t parkinsons-detection .
docker run -p 7860:7860 parkinsons-detection
```

## Usage Guide

### Disease Detection Tab
1. **Record Audio**: Click "Record" tab and record 5-10 seconds of sustained vowel sound
   - OR **Upload Audio**: Click "Upload" tab and upload a WAV/MP3/FLAC file
2. **Select Noise Reduction**: Choose light/medium/heavy based on recording quality
3. **Click "Analyze Audio"**: System will process and display results
4. **Review Results**: See detection outcome and extracted features

### Stage Classification Tab
1. **Automatic Analysis**: After disease detection, results populate automatically
2. **View Stage Information**: 
   - Stage number and severity level
   - Clinical description
   - Symptoms and characteristics
   - Medical recommendations
   - Weighted voting analysis
3. **Review Visual References**: Clinical images for the diagnosed stage
4. **Examine Feature Contributions**: See how each feature voted for the stage

## Technical Details

### Voice Biomarkers (15 Features)
1. **Jitter (4)**: Frequency variation measures
   - Jitter(%), Jitter:RAP, Jitter:PPQ5, Jitter:DDP
2. **Shimmer (6)**: Amplitude variation measures
   - Shimmer, Shimmer(dB), Shimmer:APQ3, APQ5, APQ11, DDA
3. **Harmonicity (2)**: Voice quality indicators
   - NHR, HNR
4. **Nonlinear (3)**: Complexity measures
   - RPDE, DFA, PPE

### Processing Pipeline
1. **DC Offset Removal**: Eliminates bias
2. **Noise Reduction**: Spectral subtraction
3. **Bandpass Filtering**: 80-4000 Hz
4. **RMS Normalization**: Consistent amplitude
5. **Voice Activity Detection**: Removes silence
6. **Feature Extraction**: 15 biomarkers

### Stage Classification
- **Weighted Voting System**: Features have different weights (1-5)
- **Four Stages**: Early (1), Mild (2), Moderate (3), Severe (4)
- **Confidence Score**: Percentage based on voting weights

## Troubleshooting

### Audio Recording Issues
- **Microphone not detected**: Check browser permissions
- **No sound recorded**: Verify microphone is working in system settings
- **Poor quality**: Use quiet environment, speak clearly

### Analysis Errors
- **"Analysis Error"**: Ensure audio is at least 3 seconds long
- **Feature extraction fails**: Check audio file format and quality
- **Stage classification error**: Complete disease detection first

### Model Loading
- **Model not found**: Verify `MODEL_DIR` path is correct
- **Demo mode activated**: Place model files in correct directory

## API Reference

### Main Functions

#### `detect_disease(audio_input, noise_reduction)`
Analyzes audio for Parkinson's Disease detection.
- **Parameters**:
  - `audio_input`: Audio file path or numpy array
  - `noise_reduction`: "light" | "medium" | "heavy"
- **Returns**: HTML result, DataFrame of features

#### `classify_stage(audio_input)`
Classifies disease progression stage.
- **Parameters**:
  - `audio_input`: Audio file path or numpy array
- **Returns**: HTML result, DataFrame, images

## Performance Optimization

### For Production
1. **Use GPU**: Install CUDA-enabled libraries for faster processing
2. **Cache Models**: Keep models in memory
3. **Batch Processing**: Process multiple files sequentially
4. **Optimize Audio**: Downsample to 16kHz for consistency

## Security Considerations

1. **Patient Data**: Ensure HIPAA compliance for medical data
2. **File Upload**: Validate file types and sizes
3. **Authentication**: Add user authentication for production
4. **Data Storage**: Encrypt stored audio files
5. **Logging**: Implement audit logging for medical use

## License
Ensure compliance with medical device regulations in your jurisdiction.

## Support
For issues or questions, consult the documentation or contact support.

## Version History
- **v1.0.0**: Initial Gradio conversion with full feature parity
  - Two-tab interface (Disease Detection + Stage Classification)
  - 15 biomarker analysis
  - Weighted voting stage classification
  - Professional medical UI
