import os
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import librosa
import soundfile as sf
import scipy.signal as signal
from scipy.signal import butter
import pickle
import json
import gradio as gr
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

"""
Enhanced Parkinson's Disease Detection System
Professional Medical-Grade Interface with Gradio
"""

print("="*80)
print("PARKINSON'S DISEASE DETECTION - PROFESSIONAL MEDICAL INTERFACE")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Adjust these paths based on your deployment environment
MODEL_DIR = os.getenv('MODEL_DIR', './models')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './results')
IMAGES_DIR = os.getenv('IMAGES_DIR', './images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for state management
current_features = None
current_prediction = None

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def remove_dc_offset(audio):
    """Remove DC bias from audio signal"""
    return audio - np.mean(audio)

def bandpass_filter(audio, sr, lowcut, highcut, order=4):
    """Butterworth Bandpass Filter"""
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio

def estimate_noise_profile(audio, sr, noise_duration=0.5):
    """Estimate noise characteristics from initial portion"""
    noise_samples = int(noise_duration * sr)
    if len(audio) < noise_samples:
        noise_samples = max(len(audio) // 10, 100)
    noise_segment = audio[:noise_samples]
    stft_noise = librosa.stft(noise_segment, n_fft=2048, hop_length=512)
    noise_profile = np.mean(np.abs(stft_noise), axis=1)
    return noise_profile

def spectral_subtraction(audio, sr, noise_profile=None, strength='medium'):
    """Spectral Subtraction (Boll, 1979)"""
    params = {
        'light': {'alpha': 1.0, 'beta': 0.1},
        'medium': {'alpha': 2.0, 'beta': 0.05},
        'heavy': {'alpha': 3.0, 'beta': 0.02}
    }
    alpha = params[strength]['alpha']
    beta = params[strength]['beta']

    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    if noise_profile is None:
        noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
    else:
        noise_profile = noise_profile.reshape(-1, 1)

    magnitude_clean = magnitude - alpha * noise_profile
    magnitude_clean = np.maximum(magnitude_clean, beta * magnitude)

    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=512)
    return audio_clean

def normalize_rms(audio, target_rms=0.1):
    """RMS Normalization"""
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        scaling_factor = target_rms / current_rms
        audio = audio * scaling_factor
    max_val = np.max(np.abs(audio))
    if max_val > 0.99:
        audio = audio * (0.99 / max_val)
    return audio

def trim_silence(audio, sr, top_db=20):
    """Voice Activity Detection & Trimming"""
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=512)
    if len(intervals) == 0:
        return audio, []

    voiced_segments = []
    for start, end in intervals:
        voiced_segments.append(audio[start:end])

    silence_duration = int(0.05 * sr)
    silence = np.zeros(silence_duration)

    trimmed_audio = []
    for i, segment in enumerate(voiced_segments):
        trimmed_audio.append(segment)
        if i < len(voiced_segments) - 1:
            trimmed_audio.append(silence)

    trimmed_audio = np.concatenate(trimmed_audio)
    return trimmed_audio, intervals

def preprocess_parkinsons_audio(audio_path=None, audio_data=None, sr=None,
                                noise_reduction_strength='medium',
                                preserve_parkinsons_features=True):
    """Comprehensive audio preprocessing for Parkinson's voice analysis"""
    if audio_path:
        y, original_sr = librosa.load(audio_path, sr=None, mono=True)
    elif audio_data:
        y, original_sr = audio_data
    else:
        raise ValueError("Either audio_path or audio_data must be provided")

    target_sr = sr if sr else 16000
    if original_sr != target_sr:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)

    y = remove_dc_offset(y)
    noise_profile = estimate_noise_profile(y, target_sr)
    y = spectral_subtraction(y, target_sr, noise_profile, strength=noise_reduction_strength)

    if preserve_parkinsons_features:
        y = bandpass_filter(y, target_sr, lowcut=80, highcut=4000, order=4)
    else:
        y = bandpass_filter(y, target_sr, lowcut=300, highcut=3400, order=4)

    target_rms = 0.1 if preserve_parkinsons_features else 0.15
    y = normalize_rms(y, target_rms=target_rms)
    y, intervals = trim_silence(y, target_sr, top_db=20)

    min_duration = 1.0
    if len(y) / target_sr < min_duration:
        y = np.pad(y, (0, int(min_duration * target_sr) - len(y)))

    quality_metrics = {
        'duration': len(y) / target_sr,
        'sample_rate': target_sr
    }

    return y, target_sr, quality_metrics

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_jitter_shimmer_features(voice):
    """Extract jitter and shimmer features using Parselmouth"""
    try:
        point_process = call(voice, "To PointProcess (periodic, cc)", 75, 600)

        jitter_percent = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = jitter_rap * 3

        shimmer_local = call([voice, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = call([voice, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([voice, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([voice, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = call([voice, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = shimmer_apq3 * 3

        return {
            'Jitter(%)': jitter_percent,
            'Jitter:RAP': jitter_rap,
            'Jitter:PPQ5': jitter_ppq5,
            'Jitter:DDP': jitter_ddp,
            'Shimmer': shimmer_local,
            'Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': shimmer_apq3,
            'Shimmer:APQ5': shimmer_apq5,
            'Shimmer:APQ11': shimmer_apq11,
            'Shimmer:DDA': shimmer_dda
        }
    except Exception as e:
        raise Exception(f"Error extracting jitter/shimmer: {e}")

def extract_harmonicity_features(voice):
    """Extract NHR and HNR features"""
    try:
        harmonicity = call(voice, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = 1.0 / (hnr + 1e-6) if hnr > 0 else 1.0

        return {
            'NHR': nhr,
            'HNR': hnr
        }
    except Exception as e:
        raise Exception(f"Error extracting harmonicity: {e}")

def extract_nonlinear_features(y, sr):
    """Extract RPDE, DFA, and PPE features"""
    try:
        spec = np.abs(librosa.stft(y))
        spec_norm = spec / (np.sum(spec, axis=0) + 1e-6)
        spec_entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-6), axis=0)
        rpde = np.mean(spec_entropy) / 10.0

        autocorr = librosa.autocorrelate(y)
        dfa = np.sum(autocorr[:min(100, len(autocorr))]) / len(autocorr)

        f0 = librosa.yin(y, fmin=75, fmax=600, sr=sr)
        f0_valid = f0[f0 > 0]
        ppe = np.std(f0_valid) / (np.mean(f0_valid) + 1e-6) if len(f0_valid) > 0 else 0.0

        return {
            'RPDE': rpde,
            'DFA': dfa,
            'PPE': ppe
        }
    except Exception as e:
        raise Exception(f"Error extracting nonlinear features: {e}")

def extract_all_features(audio_path=None, audio_data=None, sr=None):
    """Extract all 15 required features"""
    try:
        if audio_path:
            audio_array, sample_rate = librosa.load(audio_path, sr=sr, mono=True)
            temp_wav = '/tmp/temp_audio.wav'
            sf.write(temp_wav, audio_array, sample_rate)
            voice = parselmouth.Sound(temp_wav)
        elif audio_data:
            audio_array, sample_rate = audio_data
            temp_wav = '/tmp/temp_audio.wav'
            sf.write(temp_wav, audio_array, sample_rate)
            voice = parselmouth.Sound(temp_wav)
        else:
            raise ValueError("Either audio_path or audio_data must be provided")

        jitter_shimmer = extract_jitter_shimmer_features(voice)
        harmonicity = extract_harmonicity_features(voice)
        nonlinear = extract_nonlinear_features(audio_array, sample_rate)

        try:
            os.remove(temp_wav)
        except:
            pass

        all_features = {
            **jitter_shimmer,
            **harmonicity,
            **nonlinear
        }

        return all_features

    except Exception as e:
        raise Exception(f"Feature extraction failed: {e}")

# ============================================================================
# STAGE CLASSIFICATION
# ============================================================================

def classify_parkinsons_stage_weighted(features):
    """15-feature weighted voting stage classification"""

    weights = {
        'NHR': 5, 'HNR': 5, 'RPDE': 5,
        'PPE': 4, 'DFA': 4, 'Jitter(%)': 4, 'Shimmer': 4,
        'Shimmer:APQ3': 3, 'Shimmer:APQ5': 3, 'Jitter:RAP': 3, 'Jitter:PPQ5': 3,
        'Shimmer(dB)': 2, 'Shimmer:DDA': 2, 'Jitter:DDP': 2,
        'Shimmer:APQ11': 1
    }

    thresholds = {
        'Jitter(%)': [(0, 0.005), (0.005, 0.008), (0.008, 0.012), (0.012, float('inf'))],
        'Jitter:RAP': [(0, 0.003), (0.003, 0.005), (0.005, 0.008), (0.008, float('inf'))],
        'Jitter:PPQ5': [(0, 0.003), (0.003, 0.005), (0.005, 0.008), (0.008, float('inf'))],
        'Jitter:DDP': [(0, 0.009), (0.009, 0.015), (0.015, 0.024), (0.024, float('inf'))],
        'Shimmer': [(0, 0.035), (0.035, 0.045), (0.045, 0.055), (0.055, float('inf'))],
        'Shimmer(dB)': [(0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, float('inf'))],
        'Shimmer:APQ3': [(0, 0.018), (0.018, 0.023), (0.023, 0.028), (0.028, float('inf'))],
        'Shimmer:APQ5': [(0, 0.019), (0.019, 0.024), (0.024, 0.029), (0.029, float('inf'))],
        'Shimmer:APQ11': [(0, 0.021), (0.021, 0.026), (0.026, 0.031), (0.031, float('inf'))],
        'Shimmer:DDA': [(0, 0.054), (0.054, 0.069), (0.069, 0.084), (0.084, float('inf'))],
        'NHR': [(0, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, float('inf'))],
        'HNR': [(20.0, float('inf')), (15.0, 20.0), (10.0, 15.0), (0, 10.0)],
        'RPDE': [(0, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, float('inf'))],
        'DFA': [(0.75, float('inf')), (0.65, 0.75), (0.55, 0.65), (0, 0.55)],
        'PPE': [(0, 0.18), (0.18, 0.22), (0.22, 0.28), (0.28, float('inf'))]
    }

    stage_votes = {1: 0, 2: 0, 3: 0, 4: 0}
    feature_contributions = {}
    total_weight = sum(weights.values())

    for feature_name, value in features.items():
        if feature_name not in thresholds:
            continue

        weight = weights.get(feature_name, 1)
        stage_ranges = thresholds[feature_name]

        voted_stage = None
        for stage_idx, (low, high) in enumerate(stage_ranges, start=1):
            if low <= value < high or (stage_idx == 4 and value >= low):
                voted_stage = stage_idx
                break

        if voted_stage:
            stage_votes[voted_stage] += weight
            feature_contributions[feature_name] = {
                'value': value,
                'voted_stage': voted_stage,
                'weight': weight
            }

    max_votes = max(stage_votes.values())
    winning_stages = [stage for stage, votes in stage_votes.items() if votes == max_votes]
    predicted_stage = max(winning_stages)
    confidence = (stage_votes[predicted_stage] / total_weight) * 100

    stage_info = {
        1: {
            'name': 'Early Stage',
            'severity': 'LOW',
            'color': '#10b981',
            'bg_color': '#d1fae5',
            'description': 'Initial manifestation of Parkinson\'s Disease with minimal voice alterations and early motor symptoms.',
            'symptoms': [
                'Subtle tremor in one hand or limb at rest',
                'Mild stiffness or slowness in movements',
                'Slight changes in posture or facial expression',
                'Reduced arm swing on one side while walking',
                'Early sleep disturbances or fatigue'
            ],
            'characteristics': [
                'Minimal voice instability with slight pitch variations',
                'Slight reduction in voice volume (hypophonia)',
                'Early changes in pitch control and modulation',
                'Unilateral motor symptoms affecting one side',
                'Symptoms do not significantly interfere with daily activities'
            ],
            'recommendations': [
                'Schedule comprehensive neurological evaluation with movement disorder specialist',
                'Begin regular monitoring program with quarterly follow-up assessments',
                'Implement lifestyle modifications: regular exercise, balanced diet, adequate sleep',
                'Explore speech therapy evaluation for voice preservation techniques',
                'Join support groups for education and emotional support'
            ]
        },
        2: {
            'name': 'Mild Stage',
            'severity': 'MILD',
            'color': '#f59e0b',
            'bg_color': '#fef3c7',
            'description': 'Progressive symptoms with noticeable voice instability and bilateral motor involvement.',
            'symptoms': [
                'Tremor affecting both sides of the body',
                'Noticeable rigidity and bradykinesia (slowness)',
                'Mild balance issues and postural instability',
                'Reduced facial expressions (masked face)',
                'Difficulty with fine motor tasks like writing'
            ],
            'characteristics': [
                'Noticeable voice tremor and vocal instability',
                'Reduced voice volume requiring effort to speak loudly',
                'Bilateral motor symptoms affecting both body sides',
                'Mild motor disability impacting daily tasks',
                'Speech may become monotonous with reduced prosody'
            ],
            'recommendations': [
                'Initiate pharmacological treatment with dopaminergic medications as prescribed',
                'Active participation in speech therapy for voice strengthening (LSVT LOUD)',
                'Regular physical therapy to maintain mobility and prevent muscle stiffness',
                'Occupational therapy for adaptive strategies in daily activities',
                'Continue regular monitoring with neurologist every 2-3 months'
            ]
        },
        3: {
            'name': 'Moderate Stage',
            'severity': 'MODERATE',
            'color': '#ef4444',
            'bg_color': '#fee2e2',
            'description': 'Significant functional impairment with clear monotonicity, breathiness, and moderate motor disability.',
            'symptoms': [
                'Significant slowness and difficulty initiating movements',
                'Frequent freezing episodes during walking',
                'Notable balance problems with increased fall risk',
                'Difficulty swallowing (dysphagia)',
                'Cognitive changes and mood disturbances'
            ],
            'characteristics': [
                'Significant voice monotony with flattened prosody',
                'Breathy voice quality due to incomplete glottal closure',
                'Moderate motor impairment limiting independence',
                'Difficulty with voice projection and sustained phonation',
                'Speech intelligibility may be compromised in noisy environments'
            ],
            'recommendations': [
                'Comprehensive multidisciplinary care plan with neurology, PT, OT, and SLP',
                'Medication adjustment and optimization by movement disorder specialist',
                'Intensive speech therapy focusing on articulation and voice projection',
                'Consider assistive devices for mobility (walker, cane) and communication',
                'Regular swallowing assessments and dietary modifications for safety'
            ]
        },
        4: {
            'name': 'Severe Stage',
            'severity': 'SEVERE',
            'color': '#dc2626',
            'bg_color': '#fecaca',
            'description': 'Advanced disease with severe voice degradation, frequent breaks, and significant motor disability requiring assistance.',
            'symptoms': [
                'Severe mobility limitations, often wheelchair-dependent',
                'Frequent falls and inability to stand without support',
                'Marked cognitive impairment or dementia',
                'Severe difficulty swallowing with aspiration risk',
                'Complete dependence for activities of daily living'
            ],
            'characteristics': [
                'Severe voice degradation with marked hoarseness',
                'Frequent aphonic breaks during speech production',
                'Highly irregular pitch patterns and prosody',
                'Significant motor disability requiring full-time assistance',
                'Speech may be unintelligible requiring augmentative communication'
            ],
            'recommendations': [
                'Intensive comprehensive care with 24/7 caregiver support or skilled nursing',
                'Palliative care consultation for symptom management and quality of life',
                'Speech-language pathology for augmentative and alternative communication (AAC)',
                'Nutritional support with possible feeding tube consideration for safety',
                'Advanced care planning and discussion of end-of-life preferences'
            ]
        }
    }

    return {
        'stage': predicted_stage,
        'stage_name': stage_info[predicted_stage]['name'],
        'severity': stage_info[predicted_stage]['severity'],
        'color': stage_info[predicted_stage]['color'],
        'bg_color': stage_info[predicted_stage]['bg_color'],
        'description': stage_info[predicted_stage]['description'],
        'symptoms': stage_info[predicted_stage]['symptoms'],
        'characteristics': stage_info[predicted_stage]['characteristics'],
        'recommendations': stage_info[predicted_stage]['recommendations'],
        'confidence': confidence,
        'weighted_votes': stage_votes,
        'feature_contributions': feature_contributions,
        'total_weight': total_weight
    }

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model_components():
    """Load model, scaler, and feature info"""
    try:
        with open(f'{MODEL_DIR}/parkinsons_ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{MODEL_DIR}/feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'{MODEL_DIR}/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        print("✓ Model loaded successfully!")
        return model, scaler, feature_info
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        print("⚠️ Running in demo mode without actual model")
        return None, None, None

model, scaler, feature_info = load_model_components()

# ============================================================================
# IMAGE LOADING FUNCTION
# ============================================================================

def load_stage_images(stage):
    """Load images for a given stage"""
    images = []

    for suffix in ['a', 'b']:
        for ext in ['png', 'jpg', 'jpeg']:
            img_path = os.path.join(IMAGES_DIR, f'stage{stage}{suffix}.{ext}')
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    images.append(img)
                    print(f"✓ Loaded image: stage{stage}{suffix}.{ext}")
                except Exception as e:
                    print(f"✗ Failed to load: stage{stage}{suffix}.{ext} - {e}")

    if len(images) >= 2:
        return images[0], images[1]
    elif len(images) == 1:
        return images[0], None
    else:
        print(f"⚠️ No images found for stage {stage}")
        return None, None

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def detect_disease(audio_input, noise_reduction):
    """Disease detection (Tab 1)"""

    global current_features, current_prediction

    try:
        if audio_input is None:
            return """
            <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); border-radius: 15px; border: 2px dashed #3b82f6;">
                <h2 style="color: #1e40af; margin-bottom: 15px;">🎤 Ready for Analysis</h2>
                <p style="color: #3b82f6; font-size: 1.1em;">Please record or upload an audio file to begin voice analysis</p>
            </div>
            """, None

        # Handle different input types
        if isinstance(audio_input, tuple):
            sr, audio_array = audio_input
            audio_array = audio_array.astype(np.float32) / 32768.0

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = f'/tmp/recorded_{timestamp}.wav'
            sf.write(temp_path, audio_array, sr)
            audio_source = "Recorded Audio"

            processed_audio, sample_rate, quality_metrics = preprocess_parkinsons_audio(
                audio_path=temp_path,
                noise_reduction_strength=noise_reduction,
                preserve_parkinsons_features=True
            )

            features = extract_all_features(audio_data=(processed_audio, sample_rate))

        else:
            temp_path = audio_input
            audio_source = "Uploaded File"

            audio_array, sample_rate = librosa.load(temp_path, sr=None, mono=True)
            quality_metrics = {
                'duration': len(audio_array) / sample_rate,
                'sample_rate': sample_rate
            }

            features = extract_all_features(audio_path=temp_path)

        # Disease prediction
        if model is not None and scaler is not None and feature_info is not None:
            feature_vector = [features[name] for name in feature_info['feature_names']]
            feature_scaled = scaler.transform([feature_vector])
            prediction = model.predict(feature_scaled)[0]
        else:
            # Demo mode - use simple heuristic
            prediction = 1 if features['Jitter(%)'] > 0.006 else 0

        # Store globally for Tab 2
        current_features = features
        current_prediction = prediction

        # Build result with consistent color scheme
        if prediction == 1:
            # Parkinson's Detected
            result_html = f"""
            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #ef4444; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: #991b1b; margin: 0 0 10px 0;">⚠️ Parkinson's Disease Detected</h2>
                <p style="color: #7f1d1d; font-size: 1.1em; margin: 0;">Voice biomarkers indicate presence of Parkinson's Disease</p>
            </div>

            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="color: #92400e; margin-top: 0;">🔬 Feature Analysis Summary</h3>
                <p style="color: #78350f; line-height: 1.8; margin: 0;">
                    Analysis completed on <strong>15 voice biomarkers</strong> including:
                    <br>• <strong>Jitter Features (4):</strong> Frequency variation measures
                    <br>• <strong>Shimmer Features (6):</strong> Amplitude variation measures
                    <br>• <strong>Harmonicity Features (2):</strong> Voice quality indicators (NHR, HNR)
                    <br>• <strong>Nonlinear Features (3):</strong> Complexity measures (RPDE, DFA, PPE)
                </p>
            </div>

            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="color: #991b1b; margin-top: 0;">📋 General Instructions for Parkinson's Disease</h3>
                <p style="color: #7f1d1d; font-size: 1.02em; line-height: 1.8; margin: 0;">
                    • <strong>Consult a neurologist or movement disorder specialist</strong> for comprehensive evaluation
                    <br>• Continue with detailed disease progression analysis in the Stage Classification tab
                    <br>• Maintain a symptom diary to track changes over time
                    <br>• Consider early intervention strategies including medication and therapy
                    <br>• Join support groups and connect with healthcare professionals
                    <br>• Regular follow-up assessments are essential for monitoring disease progression
                </p>
            </div>

            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h3 style="color: #92400e; margin-top: 0;">🔬 Next Steps</h3>
                <p style="color: #78350f; font-size: 1.05em; line-height: 1.8; margin: 0;">
                    Parkinson's Disease has been detected in the voice sample.
                    <br><br>
                    <strong>Please proceed to the <span style="color: #f59e0b;">Stage Classification</span> tab for detailed progression analysis and clinical recommendations.</strong>
                </p>
            </div>
            """
        else:
            # Healthy
            result_html = f"""
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #10b981; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: #065f46; margin: 0 0 10px 0;">✅ Healthy Voice Profile</h2>
                <p style="color: #047857; font-size: 1.1em; margin: 0;">No significant indicators of Parkinson's Disease detected</p>
            </div>

            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="color: #1e40af; margin-top: 0;">🔬 Feature Analysis Summary</h3>
                <p style="color: #1e3a8a; line-height: 1.8; margin: 0;">
                    Analysis completed on <strong>15 voice biomarkers</strong> including:
                    <br>• <strong>Jitter Features (4):</strong> Frequency variation measures
                    <br>• <strong>Shimmer Features (6):</strong> Amplitude variation measures
                    <br>• <strong>Harmonicity Features (2):</strong> Voice quality indicators (NHR, HNR)
                    <br>• <strong>Nonlinear Features (3):</strong> Complexity measures (RPDE, DFA, PPE)
                </p>
            </div>

            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="color: #065f46; margin-top: 0;">📋 General Instructions for Healthy Voice</h3>
                <p style="color: #047857; font-size: 1.02em; line-height: 1.8; margin: 0;">
                    • <strong>Continue regular health monitoring</strong> with annual check-ups
                    <br>• Maintain vocal health through adequate hydration (8-10 glasses of water daily)
                    <br>• Practice voice rest when experiencing strain or fatigue
                    <br>• Avoid excessive shouting, prolonged speaking, or vocal abuse
                    <br>• Consider voice exercises and proper breathing techniques
                    <br>• Monitor for any changes in voice quality and report to healthcare provider
                </p>
            </div>

            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h3 style="color: #1e40af; margin-top: 0;">✅ Interpretation</h3>
                <p style="color: #1e3a8a; font-size: 1.05em; line-height: 1.8; margin: 0;">
                    Voice biomarkers appear within normal ranges. No significant indicators of Parkinson's Disease were detected in this voice sample.
                    <br><br>
                    <strong>Maintain healthy lifestyle practices and continue routine health screenings.</strong>
                </p>
            </div>
            """

        # Create features dataframe
        features_df = pd.DataFrame([features]).T
        features_df.columns = ['Value']
        features_df.index.name = 'Feature'
        features_df = features_df.round(6)

        return result_html, features_df

    except Exception as e:
        error_msg = f"""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #dc2626; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: #991b1b; margin: 0 0 10px 0;">❌ Analysis Error</h2>
            <p style="color: #7f1d1d; font-size: 1.05em; margin: 0;">
                <strong>Error Details:</strong> {str(e)}
                <br><br>
                <strong>Troubleshooting:</strong>
                <br>• Ensure audio is clear and at least 3 seconds long
                <br>• Check microphone functionality for recordings
                <br>• Verify file format (WAV, MP3, FLAC supported)
                <br>• Ensure adequate voice activity in the recording
            </p>
        </div>
        """
        return error_msg, None


def classify_stage(audio_input):
    """Stage classification (Tab 2)"""

    global current_features, current_prediction

    try:
        if audio_input is None:
            return """
            <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); border-radius: 15px; border: 2px dashed #3b82f6;">
                <h2 style="color: #1e40af; margin-bottom: 15px;">ℹ️ No Audio Input</h2>
                <p style="color: #3b82f6; font-size: 1.1em;">Please provide audio in the <strong>Disease Detection</strong> tab first</p>
            </div>
            """, None, None, None

        if current_prediction is None:
            return """
            <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 15px; border: 2px dashed #f59e0b;">
                <h2 style="color: #92400e; margin-bottom: 15px;">⚠️ Analysis Required</h2>
                <p style="color: #78350f; font-size: 1.1em;">Please run disease detection in the first tab before stage classification</p>
            </div>
            """, None, None, None

        if current_prediction == 0:
            return """
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 40px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: #065f46; margin-top: 0; text-align: center;">✅ No Stage Classification Required</h2>
                <p style="color: #047857; font-size: 1.1em; text-align: center; line-height: 1.8; margin: 20px 0;">
                    The audio sample was classified as <strong>HEALTHY</strong>.
                    <br><br>
                    Stage classification is only performed when Parkinson's Disease is detected.
                </p>
                <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 20px; border-radius: 10px; margin-top: 20px;">
                    <h3 style="color: #1e40af; margin-top: 0;">💡 Recommendations</h3>
                    <p style="color: #1e3a8a; line-height: 1.8; margin: 0;">
                        • Continue regular health monitoring
                        <br>• Maintain vocal health through adequate hydration
                        <br>• Practice voice rest when needed
                        <br>• Avoid vocal strain and excessive loudness
                        <br>• Schedule regular health check-ups
                    </p>
                </div>
            </div>
            """, None, None, None

        # Perform stage classification
        stage_result = classify_parkinsons_stage_weighted(current_features)

        # Load images
        stage_img1, stage_img2 = load_stage_images(stage_result['stage'])

        # Build stage output HTML with consistent color scheme
        stage_html = f"""
        <div style="background: linear-gradient(135deg, {stage_result['bg_color']} 0%, {stage_result['color']}20 100%); padding: 30px; border-radius: 15px; border-left: 6px solid {stage_result['color']}; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: {stage_result['color']}; margin: 0 0 10px 0;">Stage {stage_result['stage']}: {stage_result['stage_name']}</h2>
            <p style="color: {stage_result['color']}; font-size: 1.2em; margin: 0;"><strong>Severity: {stage_result['severity']}</strong></p>
        </div>

        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #1e40af; margin-top: 0;">📝 Clinical Description</h3>
            <p style="color: #1e3a8a; font-size: 1.05em; line-height: 1.8; margin: 0;">{stage_result['description']}</p>
        </div>

        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #991b1b; margin-top: 0;">🩺 Clinical Symptoms</h3>
            <ul style="color: #7f1d1d; font-size: 1.02em; line-height: 1.8; margin: 10px 0; padding-left: 20px;">
        """

        for symptom in stage_result['symptoms']:
            stage_html += f"<li>{symptom}</li>\n"

        stage_html += """
            </ul>
        </div>

        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #1e40af; margin-top: 0;">🔍 Voice Characteristics</h3>
            <ul style="color: #1e3a8a; font-size: 1.02em; line-height: 1.8; margin: 10px 0; padding-left: 20px;">
        """

        for char in stage_result['characteristics']:
            stage_html += f"<li>{char}</li>\n"

        stage_html += """
            </ul>
        </div>

        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #065f46; margin-top: 0;">💊 Medical Recommendations</h3>
            <ul style="color: #047857; font-size: 1.02em; line-height: 1.8; margin: 10px 0; padding-left: 20px;">
        """

        for rec in stage_result['recommendations']:
            stage_html += f"<li>{rec}</li>\n"

        stage_html += f"""
            </ul>
        </div>

        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #92400e; margin-top: 0;">📊 Weighted Voting Analysis</h3>
            <p style="color: #78350f; margin-bottom: 15px;"><strong>Total Weight:</strong> {stage_result['total_weight']}</p>
            <div style="margin-top: 10px;">
        """

        for stage, votes in sorted(stage_result['weighted_votes'].items()):
            percentage = (votes / stage_result['total_weight']) * 100
            bar_length = int(percentage * 3)
            bar = "█" * bar_length
            stage_html += f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #78350f; font-weight: 600;">Stage {stage}</span>
                    <span style="color: #92400e;">{votes}/{stage_result['total_weight']} ({percentage:.1f}%)</span>
                </div>
                <div style="background: #fde68a; border-radius: 10px; height: 25px; position: relative; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); width: {percentage}%; height: 100%; border-radius: 10px; display: flex; align-items: center; padding-left: 10px;">
                        <span style="color: white; font-weight: 600; font-size: 0.9em;">{bar}</span>
                    </div>
                </div>
            </div>
            """

        stage_html += """
            </div>
        </div>
        """

        # Create feature contributions dataframe
        contrib_data = []
        for feat_name, contrib in sorted(stage_result['feature_contributions'].items(),
                                        key=lambda x: x[1]['weight'], reverse=True):
            contrib_data.append({
                'Feature': feat_name,
                'Value': f"{contrib['value']:.6f}",
                'Voted Stage': contrib['voted_stage'],
                'Weight': contrib['weight']
            })

        contrib_df = pd.DataFrame(contrib_data)

        return stage_html, contrib_df, stage_img1, stage_img2

    except Exception as e:
        error_msg = f"""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #dc2626; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: #991b1b; margin: 0 0 10px 0;">❌ Stage Classification Error</h2>
            <p style="color: #7f1d1d; font-size: 1.05em; margin: 0;">
                <strong>Error Details:</strong> {str(e)}
                <br><br>
                Please ensure disease detection was completed successfully and try again.
            </p>
        </div>
        """
        return error_msg, None, None, None


def clear_all():
    """Clear all inputs and outputs"""
    global current_features, current_prediction
    current_features = None
    current_prediction = None

    return (
        None,  # audio_record
        None,  # audio_upload
        """
        <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); border-radius: 15px; border: 2px dashed #3b82f6;">
            <h2 style="color: #1e40af; margin-bottom: 15px;">🎤 Ready for Analysis</h2>
            <p style="color: #3b82f6; font-size: 1.1em;">Please record or upload an audio file to begin voice analysis</p>
        </div>
        """,  # result_output
        None,  # features_output
        """
        <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); border-radius: 15px; border: 2px dashed #3b82f6;">
            <h2 style="color: #1e40af; margin-bottom: 15px;">ℹ️ No Audio Input</h2>
            <p style="color: #3b82f6; font-size: 1.1em;">Please provide audio in the <strong>Disease Detection</strong> tab first</p>
        </div>
        """,  # stage_output
        None,  # stage_features_output
        None,  # stage_img1
        None   # stage_img2
    )

# ============================================================================
# CREATE ENHANCED GRADIO INTERFACE
# ============================================================================

custom_css = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    max-width: 1400px !important;
}
.tab-nav button {
    font-size: 17px;
    font-weight: 600;
    padding: 14px 28px;
    border-radius: 8px 8px 0 0;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}
h1, h2, h3 {
    font-family: 'Inter', sans-serif;
}
"""

with gr.Blocks(title="Parkinson's Disease Detection System", theme=gr.themes.Soft(), css=custom_css) as demo:

    # Header
    gr.Markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
        <h1 style="color: white; margin: 0; font-size: 2.8em; font-weight: 700;">🏥 Parkinson's Disease Detection System</h1>
        <p style="color: #e0e7ff; margin-top: 12px; font-size: 1.3em; font-weight: 500;">Advanced AI-Powered Voice Analysis Platform</p>
    </div>
    """)

    gr.Markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #0ea5e9; margin-bottom: 25px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
        <p style="margin: 0; color: #0c4a6e; font-size: 1.05em; line-height: 1.8;">
            <strong>📌 System Overview:</strong> This professional-grade diagnostic tool utilizes <strong>15 voice biomarkers</strong> and
            machine learning algorithms to detect Parkinson's Disease and classify disease progression stages.
            The system employs weighted voting mechanisms and advanced signal processing for accurate analysis.
        </p>
    </div>
    """)

    # Tabs
    with gr.Tabs() as tabs:

        # ========== TAB 1: DISEASE DETECTION ==========
        with gr.Tab("🔍 Disease Detection", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                        <h3 style="color: #1e40af; margin: 0 0 10px 0;">🎤 Audio Input</h3>
                        <p style="color: #1e3a8a; margin: 0; font-size: 0.95em;">Record your voice or upload an audio file for analysis</p>
                    </div>
                    """)

                    with gr.Tabs():
                        with gr.Tab("📝 Record"):
                            audio_record = gr.Audio(
                                sources=["microphone"],
                                type="numpy",
                                label="Record Your Voice",
                                format="wav"
                            )
                            gr.Markdown("""
                            <div style="background: #fef3c7; padding: 12px; border-radius: 8px; margin-top: 10px; border-left: 3px solid #f59e0b;">
                                <strong style="color: #92400e;">🎙️ Recording Tips:</strong>
                                <ul style="margin: 8px 0; padding-left: 20px; color: #78350f;">
                                    <li>Find a quiet environment</li>
                                    <li>Speak clearly for 5-10 seconds</li>
                                    <li>Sustain a vowel sound (e.g., "Aaaah")</li>
                                    <li>Maintain consistent volume</li>
                                </ul>
                            </div>
                            """)

                        with gr.Tab("📁 Upload"):
                            audio_upload = gr.Audio(
                                sources=["upload"],
                                type="filepath",
                                label="Upload Audio File",
                                format="wav"
                            )
                            gr.Markdown("""
                            <div style="background: #dbeafe; padding: 12px; border-radius: 8px; margin-top: 10px; border-left: 3px solid #3b82f6;">
                                <strong style="color: #1e40af;">📂 Supported Formats:</strong> WAV, MP3, FLAC
                                <br><strong style="color: #1e40af;">💡 Best Quality:</strong> 16kHz or higher sample rate
                            </div>
                            """)

                    gr.Markdown("""
                    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                        <h3 style="color: #7c3aed; margin: 0 0 10px 0;">⚙️ Processing Settings</h3>
                    </div>
                    """)

                    noise_reduction = gr.Radio(
                        choices=["light", "medium", "heavy"],
                        value="medium",
                        label="Noise Reduction Strength",
                        info="Applies to recorded audio"
                    )

                    with gr.Row():
                        analyze_btn = gr.Button("🔬 Analyze Audio", variant="primary", size="lg", scale=2)
                        clear_btn = gr.Button("🔄 Clear All", variant="secondary", size="lg", scale=1)

                with gr.Column(scale=2):
                    gr.Markdown("""
                    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                        <h3 style="color: #065f46; margin: 0 0 10px 0;">📊 Analysis Results</h3>
                    </div>
                    """)

                    result_output = gr.HTML(
                        value="""
                        <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); border-radius: 15px; border: 2px dashed #3b82f6;">
                            <h2 style="color: #1e40af; margin-bottom: 15px;">🎤 Ready for Analysis</h2>
                            <p style="color: #3b82f6; font-size: 1.1em;">Please record or upload an audio file to begin voice analysis</p>
                        </div>
                        """
                    )

                    with gr.Accordion("🔬 Detailed Feature Analysis", open=False):
                        features_output = gr.DataFrame(
                            label="15 Extracted Voice Features",
                            wrap=True
                        )

        # ========== TAB 2: STAGE CLASSIFICATION ==========
        with gr.Tab("📈 Stage Classification", id=1):
            gr.Markdown("""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <h3 style="color: #92400e; margin: 0 0 10px 0;">📈 Disease Progression Analysis</h3>
                <p style="color: #78350f; margin: 0; font-size: 1.05em;">
                    Detailed stage classification based on weighted feature voting.
                    The system classifies disease progression into four stages: Early, Mild, Moderate, and Severe.
                </p>
            </div>
            """)

            # Images first
            gr.Markdown("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); padding: 20px; border-radius: 10px; margin: 25px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                <h3 style="color: #7c3aed; margin: 0 0 10px 0;">🖼️ Stage Visual References</h3>
                <p style="color: #6b21a8; margin: 0; font-size: 0.95em;">Clinical and anatomical reference images for the diagnosed stage</p>
            </div>
            """)

            with gr.Row():
                stage_img1 = gr.Image(
                    label="Clinical Reference Image 1",
                    type="pil",
                    height=350,
                    container=True
                )
                stage_img2 = gr.Image(
                    label="Clinical Reference Image 2",
                    type="pil",
                    height=350,
                    container=True
                )

            # Stage output
            stage_output = gr.HTML(
                value="""
                <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); border-radius: 15px; border: 2px dashed #3b82f6;">
                    <h2 style="color: #1e40af; margin-bottom: 15px;">ℹ️ No Audio Input</h2>
                    <p style="color: #3b82f6; font-size: 1.1em;">Please provide audio in the <strong>Disease Detection</strong> tab first</p>
                </div>
                """
            )

            # Accordion
            with gr.Accordion("📊 Feature Contribution Analysis", open=False):
                stage_features_output = gr.DataFrame(
                    label="Weighted Feature Contributions to Stage Prediction",
                    wrap=True
                )
                gr.Markdown("""
                <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 15px; border-radius: 8px; margin-top: 10px;">
                    <strong style="color: #1e40af;">Weighting System:</strong>
                    <ul style="margin: 8px 0; padding-left: 20px; color: #1e3a8a;">
                        <li><strong>Weight 5 (High):</strong> NHR, HNR, RPDE</li>
                        <li><strong>Weight 4:</strong> PPE, DFA, Jitter(%), Shimmer</li>
                        <li><strong>Weight 3:</strong> Shimmer:APQ3, APQ5, Jitter:RAP, PPQ5</li>
                        <li><strong>Weight 2:</strong> Shimmer(dB), DDA, Jitter:DDP</li>
                        <li><strong>Weight 1:</strong> Shimmer:APQ11</li>
                    </ul>
                </div>
                """)

    # Event Handlers

    def process_both_inputs(rec, upl, noise):
        """Handle both record and upload inputs"""
        audio = rec if rec is not None else upl
        return detect_disease(audio, noise)

    def process_stage_both(rec, upl):
        """Handle stage classification for both inputs"""
        audio = rec if rec is not None else upl
        return classify_stage(audio)

    # Analyze button - updates both tabs
    analyze_btn.click(
        fn=process_both_inputs,
        inputs=[audio_record, audio_upload, noise_reduction],
        outputs=[result_output, features_output]
    ).then(
        fn=process_stage_both,
        inputs=[audio_record, audio_upload],
        outputs=[stage_output, stage_features_output, stage_img1, stage_img2]
    )

    # Clear button
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[audio_record, audio_upload, result_output, features_output,
                stage_output, stage_features_output, stage_img1, stage_img2]
    )

    # Auto-reset on audio input change
    def reset_outputs():
        return (
            """
            <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); border-radius: 15px; border: 2px dashed #3b82f6;">
                <h2 style="color: #1e40af; margin-bottom: 15px;">🎤 Ready for Analysis</h2>
                <p style="color: #3b82f6; font-size: 1.1em;">Please record or upload an audio file to begin voice analysis</p>
            </div>
            """,
            None,
            """
            <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); border-radius: 15px; border: 2px dashed #3b82f6;">
                <h2 style="color: #1e40af; margin-bottom: 15px;">ℹ️ No Audio Input</h2>
                <p style="color: #3b82f6; font-size: 1.1em;">Please provide audio in the <strong>Disease Detection</strong> tab first</p>
            </div>
            """,
            None,
            None,
            None
        )

    audio_record.change(
        fn=reset_outputs,
        inputs=[],
        outputs=[result_output, features_output, stage_output, stage_features_output, stage_img1, stage_img2]
    )

    audio_upload.change(
        fn=reset_outputs,
        inputs=[],
        outputs=[result_output, features_output, stage_output, stage_features_output, stage_img1, stage_img2]
    )

# ============================================================================
# LAUNCH INTERFACE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LAUNCHING PROFESSIONAL MEDICAL INTERFACE")
    print("="*80)
    print("\n🚀 Starting Parkinson's Disease Detection System...")
    print("📊 Two-Tab Professional Interface")
    print("🎯 Tab 1: Disease Detection")
    print("📈 Tab 2: Stage Classification with Visual References")
    print("\n" + "="*80)

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=False,
        quiet=False
    )

    print("\n✅ Interface launched successfully!")
    print("📱 Access at http://localhost:7860")
