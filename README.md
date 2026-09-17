# Facial Emotion Recognition

A facial emotion recognition pipeline that uses handcrafted features (HOG + LBP),
a Quantum-Inspired Genetic Algorithm (QIGA) for feature selection, and a
feedforward deep neural network for classification. Deployed as a Streamlit app.

## Pipeline

1. Face detection (Haar cascade) and crop
2. Resize to 48x48 grayscale
3. Feature extraction: HOG + Local Binary Pattern (LBP) histogram
4. Standardization (`scaler.pkl`)
5. Feature selection using indices chosen by a Quantum-Inspired Genetic
   Algorithm during training (`features.pkl` — stores selected indices,
   not raw features)
6. Classification with a feedforward DNN (`emotion_model.pth`)

> **Note on "quantum":** QIGA is a *quantum-inspired* classical metaheuristic
> — it borrows probability/qubit-style representations from quantum computing
> to search the feature space more effectively, but it runs entirely on
> classical hardware. No quantum hardware or simulator is used.

## Model architecture

```
Linear(input_size, 512) -> BatchNorm1d -> ReLU -> Dropout(0.4)
Linear(512, 256)         -> BatchNorm1d -> ReLU -> Dropout(0.3)
Linear(256, 128)         -> ReLU
Linear(128, num_classes)
```

## Dataset

TODO: fill in — dataset (e.g. CK+), number of classes, split ratios.

## Results

TODO: fill in — accuracy / precision / recall / F1 from the training notebook.

## Repo structure

```
app.py                    Streamlit inference app
Final_HSW_QIGA.ipynb      Training notebook (feature extraction, QIGA, training)
emotion_model.pth         Trained model weights
scaler.pkl                Fitted StandardScaler
features.pkl              QIGA-selected feature indices
labels.pkl                Class index -> emotion label mapping
ck+/                      Dataset
Images/                   TODO: describe contents
```

## Running locally

```
pip install -r requirements.txt
streamlit run app.py
```
