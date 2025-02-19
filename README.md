# Captcha Breaking with CRNN

## Overview
A deep learning-driven approach to captcha solving, leveraging a Convolutional Recurrent Neural Network (CRNN) trained on the "Captcha Version 2 Images" dataset.

## Dataset
- **Source:** Kaggle - Captcha Version 2 Images
- **Format:** PNG captchas with alphanumeric sequences
- **Preprocessing:** Grayscale conversion, resized to (200x50)

## Model Architecture
- **Feature Extraction:** CNN layers
- **Sequence Modeling:** Bidirectional LSTM
- **Loss Function:** Connectionist Temporal Classification (CTC)
- **Optimizer:** SGD with momentum & Nesterov acceleration

### Detailed Model Summary
#### Model: "ocr_model_v1"
| Layer (type)                | Output Shape         | Param #  |
|-----------------------------|----------------------|----------|
| input_data (InputLayer)     | (None, 200, 50, 1)  | 0        |
| Conv1 (Conv2D)             | (None, 200, 50, 32) | 320      |
| pool1 (MaxPooling2D)       | (None, 100, 25, 32) | 0        |
| Conv2 (Conv2D)             | (None, 100, 25, 64) | 18496    |
| pool2 (MaxPooling2D)       | (None, 50, 12, 64)  | 0        |
| reshape (Reshape)          | (None, 50, 768)     | 0        |
| dense1 (Dense)             | (None, 50, 64)      | 49216    |
| dropout (Dropout)          | (None, 50, 64)      | 0        |
| bidirectional (Bidirectional) | (None, 50, 256) | 197632   |
| bidirectional_1 (Bidirectional) | (None, 50, 128) | 164352   |
| input_label (InputLayer)    | (None, 5)          | 0        |
| input_length (InputLayer)   | (None, 1)          | 0        |
| label_length (InputLayer)   | (None, 1)          | 0        |
| dense2 (Dense)             | (None, 50, 20)     | 2580     |
| ctc_loss (CTCLayer)        | (None, 1)          | 0        |

**Total Parameters:** 432,596  
**Trainable Parameters:** 432,596  
**Non-trainable Parameters:** 0  

## Training
- **Batch Size:** 16, **Epochs:** 50 (early stopping applied)
- **Validation Split:** 10%
- **Performance:** Achieves high accuracy in captcha recognition

## Usage
- Load the trained model for captcha inference.
- Use `decode_batch_predictions()` to extract text sequences.

## Author
Developed in TensorFlow & Keras as an OCR-based captcha solver.

