## 🚗 **Acc-Detector — Accident Detection using Deep Learning**

_A computer vision model that detects road accidents from images using transfer learning on MobileNetV2._

---

### 🧩 **Project Overview**

Acc-Detector is an AI/ML project built to classify whether an image depicts a road accident or not.  
It uses a pre-trained **MobileNetV2** model fine-tuned on a custom dataset organized into `Accident` and `Non-Accident` categories.

The model can later be extended to process **video feeds** for real-time accident detection.

---

### 📂 **Dataset Structure**

Your dataset should follow this format:

`data/ ├── train/ │   ├── Accident/ │   └── Non Accident/ ├── val/ │   ├── Accident/ │   └── Non Accident/ └── test/     ├── Accident/     └── Non Accident/`

---

### ⚙️ **How It Works**

1. **Preprocessing** — Images are normalized and augmented for better generalization.
    
2. **Transfer Learning** — MobileNetV2 (pretrained on ImageNet) is used as a feature extractor.
    
3. **Fine-Tuning** — Top layers are retrained on the accident dataset.
    
4. **Binary Classification** — Outputs `Accident` or `Non-Accident` with a sigmoid activation.
    

---

### 🧠 **Tech Stack**

- **Python 3.10+**
    
- **TensorFlow / Keras**
    
- **NumPy, Matplotlib**
    
- **Google Colab** (for training)
    
- **Git & GitHub** (for version control)
    

---

### 🚀 **Quick Start**

#### 1. Clone the repository

`git clone https://github.com/Vishmayraj/Acc-Detector.git cd Acc-Detector`

#### 2. Install dependencies

`pip install -r requirements.txt`

#### 3. Load your trained model

`from tensorflow.keras.models import load_model model = load_model('accident_detector.keras')`

#### 4. Make a prediction

`import numpy as np from tensorflow.keras.preprocessing import image  img = image.load_img('sample.jpg', target_size=(224, 224)) x = image.img_to_array(img) x = np.expand_dims(x, axis=0) / 255.0  pred = model.predict(x) print("🚗 Accident Detected" if pred[0] > 0.5 else "✅ Non-Accident")`

---

### 🧾 **Results**

|Metric|Value|
|---|---|
|Training Accuracy|~97%|
|Validation Accuracy|~95%|
|Test Accuracy|~94%|

_(Results may vary depending on dataset size and fine-tuning)_

---

### 🔒 **Environment Variables**

If you use private keys or environment-specific settings, store them in a `.env` file:

`API_KEY=your_key_here`

And ensure `.env` is in your `.gitignore`.

---

### 📦 **Model Files**

|File|Description|
|---|---|
|`accident_detector.keras`|Full trained model|
|`accident_weights.h5`|Model weights only|
|`train_model.ipynb`|Colab notebook used for training|

---

### 🧭 **Future Plans**

-  Real-time accident detection from video feed
    
-  Integration with IoT camera systems
    
-  REST API for live inference
    

---

### 🧑‍💻 **Author**

**VishmayRaj**

---