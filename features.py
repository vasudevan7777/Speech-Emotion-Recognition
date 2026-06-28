import os
import librosa
import numpy as np

SAMPLE_RATE = 22050
TRIM_TOP_DB = 30
MIN_SAMPLES  = int(SAMPLE_RATE * 0.5)


def preprocess_audio(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = y - np.mean(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if len(y_trimmed) >= MIN_SAMPLES:
        y = y_trimmed
    if len(y) < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - len(y)), mode='constant')
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak
    return y


def audio_quality_check(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    warns = []
    duration_s = len(y) / sr
    peak_amp   = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    rms        = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0

    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        speech_s = len(y_trimmed) / sr
    except Exception:
        speech_s = duration_s

    if peak_amp < 0.01:
        warns.append("[QUIET] Recording too quiet - please speak louder or move closer to the microphone.")
    elif peak_amp < 0.05:
        warns.append("[LOW VOLUME] Low recording volume - results may be less accurate.")

    if speech_s < 0.8:
        warns.append("[TOO SHORT] Recording too short - please speak for at least 1-2 seconds.")
    elif duration_s > 0 and speech_s / duration_s < 0.25:
        warns.append("[SILENCE] Mostly silence detected - please speak immediately after clicking record.")

    if rms > 0.05 and peak_amp < 0.15:
        warns.append("[NOISE] Background noise detected - try recording in a quieter environment.")

    return {
        "ok":         len(warns) == 0,
        "duration_s": round(duration_s, 2),
        "speech_s":   round(speech_s,   2),
        "peak_amp":   round(peak_amp,   4),
        "rms":        round(rms,        4),
        "warnings":   warns,
    }


def extract_features(
    audio_input,
    sr: int           = SAMPLE_RATE,
    num_features: int = 549,
    duration: float   = 3.0,
    offset: float     = 0.5,
) -> np.ndarray:
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        y, sample_rate = librosa.load(audio_input, sr=sr, duration=duration, offset=offset)
    else:
        y, sample_rate = np.asarray(audio_input, dtype=np.float32), sr

    y = preprocess_audio(y, sample_rate)
    return _extract_189(y, sample_rate) if num_features == 189 else _extract_549(y, sample_rate)


def _extract_189(y: np.ndarray, sr: int) -> np.ndarray:
    mfccs         = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,          axis=0)
    chroma        = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T,               axis=0)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T,         axis=0)
    zcr           = np.mean(librosa.feature.zero_crossing_rate(y).T,                  axis=0)
    rms           = np.mean(librosa.feature.rms(y=y).T,                               axis=0)
    mel           = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)
    return np.hstack([mfccs, chroma, spec_contrast, zcr, rms, mel])


def _extract_549(y: np.ndarray, sr: int) -> np.ndarray:
    feats = []

    mfccs       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_d      = librosa.feature.delta(mfccs)
    mfcc_d2     = librosa.feature.delta(mfccs, order=2)

    feats.append(np.mean(mfccs,  axis=1))
    feats.append(np.std(mfccs,   axis=1))
    feats.append(np.mean(mfcc_d, axis=1))
    feats.append(np.std(mfcc_d,  axis=1))
    feats.append(np.mean(mfcc_d2, axis=1))
    feats.append(np.std(mfcc_d2,  axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    feats.append(np.mean(chroma, axis=1))
    feats.append(np.std(chroma,  axis=1))

    sc = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.append(np.mean(sc, axis=1))
    feats.append(np.std(sc,  axis=1))

    zcr = librosa.feature.zero_crossing_rate(y)
    feats.append(np.array([np.mean(zcr), np.std(zcr)]))

    rms = librosa.feature.rms(y=y)
    feats.append(np.array([np.mean(rms), np.std(rms)]))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    feats.append(np.mean(mel, axis=1))
    feats.append(np.std(mel,  axis=1))

    centroid  = librosa.feature.spectral_centroid(y=y,  sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y,   sr=sr)
    flatness  = librosa.feature.spectral_flatness(y=y)
    feats.append(np.array([
        np.mean(centroid),  np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff),   np.std(rolloff),
        np.mean(flatness),  np.std(flatness),
    ]))

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    voiced = pitches[pitches > 0]
    voiced_fraction = float(len(voiced)) / float(pitches.size + 1e-8)
    feats.append(np.array([
        float(np.mean(voiced)) if len(voiced) > 0 else 0.0,
        float(np.std(voiced))  if len(voiced) > 0 else 0.0,
        voiced_fraction,
    ]))

    return np.hstack(feats)
