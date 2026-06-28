import os
import librosa
import numpy as np
import joblib
import sounddevice as sd
import soundfile as sf
from colorama import Fore, Style, init
import warnings
warnings.filterwarnings('ignore')

import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

class SimpleLabelEncoder:
    def __init__(self):
        self.classes_ = None
    def fit(self, y):
        self.classes_ = np.unique(y)
        return self
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    def transform(self, y):
        mapping = {c: i for i, c in enumerate(self.classes_)}
        return np.array([mapping[x] for x in y])
    def inverse_transform(self, y):
        return np.array([self.classes_[x] for x in y])

class SimpleStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        self.n_features_in_ = X.shape[1]
        return self
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    def transform(self, X):
        return (X - self.mean_) / self.scale_

def simple_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def simple_accuracy_score(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))

def simple_classification_report(y_true, y_pred, target_names, zero_division=0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    report = f"{'emotion':15s} {'precision':10s} {'recall':10s} {'f1-score':10s} {'support':10s}\n\n"
    precisions, recalls, f1s, supports = [], [], [], []
    
    for i, name in enumerate(target_names):
        true_mask = (y_true == i)
        pred_mask = (y_pred == i)
        
        tp = np.sum(true_mask & pred_mask)
        fp = np.sum(~true_mask & pred_mask)
        fn = np.sum(true_mask & ~pred_mask)
        support = np.sum(true_mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else zero_division
        
        report += f"{name:15s} {precision:10.2f} {recall:10.2f} {f1:10.2f} {support:10d}\n"
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        
    total_support = np.sum(supports)
    macro_precision = np.mean(precisions) if len(precisions) > 0 else 0
    macro_recall = np.mean(recalls) if len(recalls) > 0 else 0
    macro_f1 = np.mean(f1s) if len(f1s) > 0 else 0
    
    weighted_precision = np.sum(np.array(precisions) * np.array(supports)) / total_support if total_support > 0 else 0
    weighted_recall = np.sum(np.array(recalls) * np.array(supports)) / total_support if total_support > 0 else 0
    weighted_f1 = np.sum(np.array(f1s) * np.array(supports)) / total_support if total_support > 0 else 0
    
    accuracy = np.mean(y_true == y_pred)
    
    report += "\n"
    report += f"{'accuracy':15s} {'':21s} {accuracy:10.2f} {total_support:10d}\n"
    report += f"{'macro avg':15s} {macro_precision:10.2f} {macro_recall:10.2f} {macro_f1:10.2f} {total_support:10d}\n"
    report += f"{'weighted avg':15s} {weighted_precision:10.2f} {weighted_recall:10.2f} {weighted_f1:10.2f} {total_support:10d}\n"
    return report

LOW_CONFIDENCE_THRESHOLD      = 0.45
MODERATE_CONFIDENCE_THRESHOLD = 0.60

class KerasEmotionModel:
    def __init__(self, scaler=None, keras_model=None, classes=None):
        self.scaler       = scaler
        self.keras_model  = keras_model
        self.classes_     = classes
        self.n_features_in_ = scaler.n_features_in_ if scaler is not None else None

    def fit(self, X, y, epochs=100, batch_size=64):
        import random as _py_random
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2

        _SEED = 42
        _py_random.seed(_SEED)
        np.random.seed(_SEED)
        tf.random.set_seed(_SEED)

        self.scaler = SimpleStandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.n_features_in_ = X.shape[1]

        np.random.seed(_SEED)
        X_noise_1 = X_scaled + np.random.normal(0, 0.015, X_scaled.shape).astype(np.float32)
        X_noise_2 = X_scaled + np.random.normal(0, 0.025, X_scaled.shape).astype(np.float32)
        X_aug = np.vstack([X_scaled, X_noise_1, X_noise_2])
        y_aug = np.tile(y, 3)

        rng = np.random.RandomState(_SEED)
        perm = rng.permutation(len(X_aug))
        X_aug, y_aug = X_aug[perm], y_aug[perm]

        num_classes   = len(np.unique(y_aug))
        self.classes_ = np.unique(y_aug)

        class_counts = np.bincount(y_aug.astype(int))
        class_weight = {
            i: float(len(y_aug)) / (num_classes * cnt)
            for i, cnt in enumerate(class_counts)
            if cnt > 0
        }

        L2 = 5e-4
        self.keras_model = Sequential([
            Input(shape=(self.n_features_in_,)),

            Dense(512, activation='relu',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(L2)),
            BatchNormalization(),
            Dropout(0.40),

            Dense(256, activation='relu',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(L2)),
            BatchNormalization(),
            Dropout(0.35),

            Dense(128, activation='relu',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(L2)),
            BatchNormalization(),
            Dropout(0.30),

            Dense(64, activation='relu',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(L2)),
            BatchNormalization(),
            Dropout(0.25),

            Dense(num_classes, activation='softmax'),
        ])

        self.keras_model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.4,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        self.keras_model.fit(
            X_aug, y_aug,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.20,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        proba = self.keras_model.predict(X_scaled, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.keras_model.predict(X_scaled, verbose=0)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        keras_path = os.path.join(model_dir, "model.keras")
        self.keras_model.save(keras_path)

        temp_model = self.keras_model
        self.keras_model = None
        joblib.dump(self, os.path.join(model_dir, "model.pkl"))
        self.keras_model = temp_model

    @classmethod
    def load(cls, model_dir):
        from tensorflow.keras.models import load_model

        wrapper = joblib.load(os.path.join(model_dir, "model.pkl"))
        wrapper.keras_model = load_model(os.path.join(model_dir, "model.keras"))
        if wrapper.scaler is not None and wrapper.scaler.n_features_in_ is not None:
            wrapper.n_features_in_ = wrapper.scaler.n_features_in_
        return wrapper


# Initialize colorama
init(autoreset=True)

# Path to RAVDESS dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Audio_Speech_Actors_01-24")

# Emotion labels based on RAVDESS filenames (3rd position in filename)
# Format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

from features import extract_features as _extract_features

# Feature extractor: MFCC + Chroma + Spectral Contrast + Zero Crossing Rate + RMS Energy
def extract_features(file_path, duration=5, offset=0.0):
    """Extract comprehensive audio features for emotion recognition."""
    return _extract_features(file_path, num_features=189, duration=duration, offset=offset)

# Load RAVDESS dataset from Actor folders
def load_dataset():
    """Load all audio files from the RAVDESS dataset (Actor folders)."""
    x, y = [], []
    
    if not os.path.exists(DATASET_PATH):
        print(Fore.RED + f"❌ Dataset not found at: {DATASET_PATH}")
        print(Fore.YELLOW + "Please ensure the RAVDESS dataset is in the correct location.")
        return np.array(x), np.array(y)
    
    actor_folders = sorted(os.listdir(DATASET_PATH))
    total_actors = len([f for f in actor_folders if os.path.isdir(os.path.join(DATASET_PATH, f))])
    
    print(Fore.CYAN + f"   Found {total_actors} actor folders")
    
    for i, actor_folder in enumerate(actor_folders):
        actor_path = os.path.join(DATASET_PATH, actor_folder)
        if os.path.isdir(actor_path):
            actor_files = 0
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if len(parts) >= 3:
                        emotion_code = parts[2]  # 3rd element is emotion
                        emotion = EMOTIONS.get(emotion_code)
                        if emotion:
                            try:
                                features = extract_features(os.path.join(actor_path, file), duration=3, offset=0.5)
                                x.append(features)
                                y.append(emotion)
                                actor_files += 1
                            except Exception as e:
                                pass
            
            progress = (i + 1) / total_actors * 100
            print(f"\r   Loading: {progress:.0f}% ({actor_folder}: {actor_files} files)", end="", flush=True)
    
    print()
    return np.array(x), np.array(y)

def train_model(x_train, y_train):
    model = KerasEmotionModel()
    model.fit(x_train, y_train, epochs=100, batch_size=64)
    return model


def record_audio(filename="live.wav", duration=5, samplerate=22050):
    print(Fore.CYAN + f"🎙 Recording for {duration} seconds... Speak now!")
    print(Fore.YELLOW + "   ", end="")
    
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    
    for i in range(duration, 0, -1):
        print(f"{i}...", end="", flush=True)
        sd.sleep(1000)
    
    sd.wait()
    print(" Done!")
    sf.write(filename, recording, samplerate)
    print(Fore.GREEN + f"✅ Recording saved as {filename}")

def predict_emotion(model, filename, label_encoder):
    features = extract_features(filename, duration=5, offset=0.0).reshape(1, -1)
    prediction_encoded = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    emotion = label_encoder.inverse_transform([prediction_encoded])[0]
    confidence = np.max(proba)
    
    all_probs = dict(zip(label_encoder.inverse_transform(model.classes_), proba))
    
    return emotion, confidence, all_probs

def adaptive_train(model, x_train, y_train, new_feature, new_label_encoded, label_encoder):
    x_train = np.vstack([x_train, new_feature])
    y_train = np.append(y_train, new_label_encoded)
    model.fit(x_train, y_train)
    print(Fore.GREEN + "✅ Model updated with new sample!")
    return model, x_train, y_train

def show_emotion_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    print(Fore.CYAN + "\n📊 Emotion Distribution in Dataset:")
    print("-" * 40)
    for emotion, count in zip(unique, counts):
        bar = "█" * (count // 5)
        print(f"  {emotion:12s}: {count:3d} samples {bar}")
    print("-" * 40)

if __name__ == "__main__":
    print(Fore.YELLOW + "\n" + "=" * 50)
    print(Fore.YELLOW + "   🎭 SPEECH EMOTION RECOGNITION SYSTEM 🎭")
    print(Fore.YELLOW + "=" * 50)
    
    print(Fore.CYAN + "\n🔄 Loading RAVDESS dataset...")
    
    x, y = load_dataset()
    
    if len(x) == 0:
        print(Fore.RED + "❌ No audio files found! Please check dataset path.")
        exit()
    
    print(Fore.GREEN + f"✅ Loaded {len(x)} audio samples")
    
    show_emotion_distribution(y)
    
    print(Fore.CYAN + "\n🧠 Encoding labels and training model...")
    label_encoder = SimpleLabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    x_train, x_test, y_train, y_test = simple_train_test_split(
        x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"   Training samples: {len(x_train)}")
    print(f"   Testing samples:  {len(x_test)}")

    model = train_model(x_train, y_train)
    print(Fore.GREEN + "\n✅ Model training completed!")
    

    y_pred = model.predict(x_test)
    accuracy = simple_accuracy_score(y_test, y_pred) * 100
    
    print(Fore.YELLOW + "\n" + "=" * 50)
    print(Fore.YELLOW + f"   🎯 MODEL ACCURACY: {accuracy:.2f}%")
    print(Fore.YELLOW + "=" * 50)
    
    print(Fore.CYAN + "\n📋 Classification Report:")
    print(simple_classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    while True:
        print(Fore.YELLOW + "\n" + "-" * 50)
        print(Fore.YELLOW + "Options:")
        print("  1. Record and predict emotion")
        print("  2. Test with existing file")
        print("  3. Exit")
        
        choice = input(Fore.WHITE + "\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            record_audio()
            emotion, conf, all_probs = predict_emotion(model, "live.wav", label_encoder)
            
            print(Fore.MAGENTA + "\n" + "=" * 50)
            print(Fore.MAGENTA + "   🎤 EMOTION PREDICTION RESULT")
            print(Fore.MAGENTA + "=" * 50)
            print(Fore.GREEN + f"   🗣 Detected Emotion : {emotion.upper()}")
            print(Fore.GREEN + f"   📊 Confidence Level : {conf * 100:.2f}%")
            
            print(Fore.CYAN + "\n   All Emotion Probabilities:")
            for emo, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                marker = " ◄" if emo == emotion else ""
                print(f"   {emo:12s}: {prob*100:5.1f}% {bar}{marker}")
            
            if conf > 0.85:
                print(Fore.CYAN + f"\n   ➕ High confidence! Add '{emotion}' to training data? (y/n): ", end="")
                if input().strip().lower() == 'y':
                    new_feat = extract_features("live.wav", duration=5, offset=0.0).reshape(1, -1)
                    new_label = label_encoder.transform([emotion])[0]
                    model, x_train, y_train = adaptive_train(
                        model, x_train, y_train, new_feat, new_label, label_encoder
                    )
                    
        elif choice == "2":
            filepath = input(Fore.WHITE + "Enter audio file path: ").strip().strip('"')
            if os.path.exists(filepath):
                emotion, conf, all_probs = predict_emotion(model, filepath, label_encoder)
                print(Fore.MAGENTA + f"\n   🎤 Detected Emotion: {emotion.upper()} ({conf*100:.2f}%)")
            else:
                print(Fore.RED + "   ❌ File not found!")
                
        elif choice == "3":
            print(Fore.GREEN + "\n👋 Goodbye!")
            break
        else:
            print(Fore.RED + "   ❌ Invalid choice!")