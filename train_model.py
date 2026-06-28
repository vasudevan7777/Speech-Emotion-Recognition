import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import joblib
from joblib import Parallel, delayed

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from features import extract_features, SAMPLE_RATE
from speech_emotion_recognition import (
    KerasEmotionModel,
    SimpleLabelEncoder,
    simple_train_test_split,
    simple_accuracy_score,
    simple_classification_report,
)


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Audio_Speech_Actors_01-24")
MODEL_DIR    = os.path.join(BASE_DIR, "models")

EMOTIONS = {
    '01': 'neutral',  '02': 'calm',    '03': 'happy',   '04': 'sad',
    '05': 'angry',    '06': 'fearful', '07': 'disgust', '08': 'surprised',
}

NUM_FEATURES = 549
N_JOBS       = -1


def augment_audio(y: np.ndarray, sr: int) -> list:
    variants = [y]

    noise_std = 0.005 * max(float(np.max(np.abs(y))), 1e-8)
    noisy = np.clip(
        y + np.random.normal(0, noise_std, len(y)).astype(np.float32),
        -1.0, 1.0
    )
    variants.append(noisy)

    try:
        variants.append(librosa.effects.time_stretch(y, rate=0.90))
    except Exception:
        pass

    try:
        variants.append(librosa.effects.time_stretch(y, rate=1.10))
    except Exception:
        pass

    return variants


def _process_file(filepath: str, emotion: str) -> list:
    results = []
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=3.0, offset=0.5)
        for y_var in augment_audio(y, sr):
            try:
                feat = extract_features(y_var, sr=sr, num_features=NUM_FEATURES)
                if feat.shape[0] == NUM_FEATURES:
                    results.append((feat, emotion))
            except Exception:
                pass
    except Exception:
        pass
    return results


def load_and_augment_dataset() -> tuple:
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        return np.array([]), np.array([])

    tasks = []
    for actor in sorted(os.listdir(DATASET_PATH)):
        actor_path = os.path.join(DATASET_PATH, actor)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if not fname.endswith('.wav'):
                continue
            parts = fname.split('-')
            if len(parts) < 3:
                continue
            emotion = EMOTIONS.get(parts[2])
            if emotion:
                tasks.append((os.path.join(actor_path, fname), emotion))

    print(f"  Found {len(tasks)} WAV files — extracting features in parallel "
          f"(N_JOBS={N_JOBS})...")

    raw_results = Parallel(n_jobs=N_JOBS, verbose=5, prefer='threads')(
        delayed(_process_file)(fp, em) for fp, em in tasks
    )

    x_list, y_list = [], []
    for batch in raw_results:
        for feat, em in batch:
            x_list.append(feat)
            y_list.append(em)

    return np.array(x_list, dtype=np.float32), np.array(y_list)

if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 65)
    print("   SPEECH EMOTION RECOGNITION — FAST TRAINING  (v4)")
    print("=" * 65)
    print("   Augmentations : noise + time-stretch ×2  (4 variants/clip)")
    print("   Feature dims  : 549  (mean+std for all banks)")
    print("   Batch size    : 64   (4× fewer steps per epoch)")
    print("   Max epochs    : 100  (EarlyStopping patience=15)")
    print("=" * 65)

    print("\n[1/4] Loading & augmenting RAVDESS dataset (parallel)...")
    X, y = load_and_augment_dataset()

    if len(X) == 0:
        print("[ERROR] No samples loaded. Check DATASET_PATH.")
        sys.exit(1)

    print(f"\n  Total feature vectors : {len(X)}")
    print(f"  Feature dimensionality: {X.shape[1]}")
    print("\n  Emotion distribution (after augmentation):")
    unique_labels, counts = np.unique(y, return_counts=True)
    for em, cnt in zip(unique_labels, counts):
        bar = '█' * (cnt // 15)
        print(f"    {em:12s}: {cnt:5d}  {bar}")

    print("\n[2/4] Encoding labels and splitting train/test (80/20)...")
    le = SimpleLabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = simple_train_test_split(
        X, y_enc, test_size=0.20, random_state=42,
    )
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test  samples: {len(X_test)}")

    print("\n[3/4] Training 512→256→128→64 MLP  (batch=64, epochs≤100)...")
    print("  (Expected: 8-15 minutes on CPU)\n")

    model = KerasEmotionModel()
    model.fit(X_train, y_train, epochs=100, batch_size=64)

    print("\n[4/4] Evaluating and saving model...")
    y_pred   = model.predict(X_test)
    accuracy = simple_accuracy_score(y_test, y_pred) * 100

    print("\n" + "=" * 65)
    print(f"   FINAL ACCURACY: {accuracy:.2f}%")
    print("=" * 65)

    print("\nClassification Report:")
    print(simple_classification_report(y_test, y_pred, le.classes_, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_DIR)
    joblib.dump(le,       os.path.join(MODEL_DIR, "encoder.pkl"))
    joblib.dump(accuracy, os.path.join(MODEL_DIR, "accuracy.pkl"))

    print(f"\n  Model saved to : {MODEL_DIR}")
    print(f"  Accuracy       : {accuracy:.2f}%")
    print("\n  Launch the app:  streamlit run app.py")
    print("=" * 65)
