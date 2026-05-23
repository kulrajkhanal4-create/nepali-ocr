from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import io

app = Flask(__name__)

# ── Character CNN Model ───────────────────────────────────────
class NepaliOCR_CNN(nn.Module):
    def __init__(self, num_classes=46):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ── CRNN Model ────────────────────────────────────────────────
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, padding=1),nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(512, 512, 3, padding=1),nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2,1)),
        )
        self.rnn = nn.LSTM(512*2, 256, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.3)
        self.fc  = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c*h)
        x, _ = self.rnn(x)
        return self.fc(x).log_softmax(2)

# ── Vocabulary ────────────────────────────────────────────────
CHARS = ['<blank>',
         'क','ख','ग','घ','ङ','च','छ','ज','झ','ञ',
         'ट','ठ','ड','ढ','ण','त','थ','द','ध','न',
         'प','फ','ब','भ','म','य','र','ल','व','श',
         'ष','स','ह','क्ष','त्र','ज्ञ',
         '०','१','२','३','४','५','६','७','८','९',
         'ा','ि','ी','ु','ू','े','ै','ो','ौ','ं','ः','्','ँ','ृ',
         'अ','आ','इ','ई','उ','ऊ','ए','ऐ','ओ','औ','अं','अः']

idx2char = {i: c for i, c in enumerate(CHARS)}

CHAR_NAMES = ['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ',
              'ट','ठ','ड','ढ','ण','त','थ','द','ध','न',
              'प','फ','ब','भ','म','य','र','ल','व','श',
              'ष','स','ह','क्ष','त्र','ज्ञ',
              '०','१','२','३','४','५','६','७','८','९']

# ── Load models ───────────────────────────────────────────────
device = torch.device('cpu')

cnn_model = NepaliOCR_CNN(num_classes=46)
cnn_model.load_state_dict(torch.load('../models/nepali_ocr_cnn.pth',
                                      map_location=device))
cnn_model.eval()

crnn_model = CRNN(num_classes=len(CHARS))
crnn_model.load_state_dict(torch.load('../models/crnn_nepali_v4.pth',
                                       map_location=device))
crnn_model.eval()
print("Both models loaded!")

FONT_PATH = '/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Bold.ttf'

# ── CTC Decoder ───────────────────────────────────────────────
def ctc_decode(output):
    pred_ids = output.argmax(1).tolist()
    chars = []
    prev  = -1
    for idx in pred_ids:
        if idx != prev and idx != 0:
            chars.append(idx2char[idx])
        prev = idx
    return ''.join(chars)

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_char', methods=['POST'])
def predict_char():
    data     = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    img      = Image.open(io.BytesIO(img_data)).convert('L').resize((32, 32))
    img_np   = np.array(img).astype(np.float32) / 255.0
    if img_np.mean() > 0.5:
        img_np = 1.0 - img_np
    tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    with torch.no_grad():
        out   = cnn_model(tensor)
        probs = out.softmax(1)[0]
        top5  = probs.topk(5)
    results = [
        {'char': CHAR_NAMES[i], 'confidence': round(probs[i].item()*100, 2)}
        for i in top5.indices.tolist()
    ]
    return jsonify({'predictions': results})

@app.route('/predict_word', methods=['POST'])
def predict_word():
    word = request.json['word']
    # Generate word image
    font    = ImageFont.truetype(FONT_PATH, size=48)
    img     = Image.new('L', (256, 64), color=255)
    draw    = ImageDraw.Draw(img)
    bbox    = draw.textbbox((0,0), word, font=font)
    w, h    = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x       = max(5, (256-w)//2)
    y       = max(5, (64-h)//2)
    draw.text((x, y), word, font=font, fill=0)

    # Convert to tensor
    img_np  = np.array(img).astype(np.float32) / 255.0
    tensor  = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    tensor  = (tensor - 0.5) / 0.5

    with torch.no_grad():
        out  = crnn_model(tensor)
        pred = ctc_decode(out[0])

    # Convert image to base64 for display
    buf = io.BytesIO()
    Image.fromarray((img_np * 255).astype(np.uint8)).save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({'prediction': pred, 'image': img_b64})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
