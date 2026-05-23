# Nepali OCR 🔤

Handwritten Devanagari Character and Word Recognition using Deep Learning.

## Models
- **CNN** — 99.30% accuracy on 46 Devanagari characters (DHCD dataset)
- **CRNN + CTC** — Word recognition without segmentation

## Dataset
- DHCD Dataset — 78,200 training images, 13,800 test images
- 46 classes (36 characters + 10 Nepali digits)

## Features
- Draw a character — CNN predicts with confidence
- Type a Nepali word — CRNN reads it directly
- Flask web app with two-tab interface

## Tech Stack
- Python, PyTorch, Flask, OpenCV
- CNN + BiLSTM + CTC Loss

## Results
| Model | Accuracy |
|-------|----------|
| CNN (characters) | 99.30% |
| CRNN (words) | 12/12 test words |

## Run
```bash
cd backend
python3 app.py
```
Open http://localhost:5000

## Author
Kulraj Khanal — Computer Engineering Lecturer & Master's Student
