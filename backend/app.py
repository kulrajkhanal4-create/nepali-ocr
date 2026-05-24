from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import io
from transformers import MarianMTModel, MarianTokenizer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Character CNN ─────────────────────────────────────────────
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
            nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ── CRNN ──────────────────────────────────────────────────────
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

idx2char   = {i: c for i, c in enumerate(CHARS)}
CHAR_NAMES = ['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ',
              'ट','ठ','ड','ढ','ण','त','थ','द','ध','न',
              'प','फ','ब','भ','म','य','र','ल','व','श',
              'ष','स','ह','क्ष','त्र','ज्ञ',
              '०','१','२','३','४','५','६','७','८','९']

# ── Load models ───────────────────────────────────────────────
device = torch.device('cpu')

cnn_model = NepaliOCR_CNN(num_classes=46)
cnn_model.load_state_dict(torch.load('../models/nepali_ocr_cnn.pth', map_location=device))
cnn_model.eval()

crnn_model = CRNN(num_classes=len(CHARS))
crnn_model.load_state_dict(torch.load('../models/crnn_nepali_v4.pth', map_location=device))
crnn_model.eval()

print("Loading translation model...")
mt_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
mt_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
print("All models loaded!")

FONT_PATH = '/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Bold.ttf'

# ── Spell corrector ───────────────────────────────────────────
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp   = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],
                                   dp[i][j-1],
                                   dp[i-1][j-1])
    return dp[m][n]

NEPALI_DICTIONARY = [
    'नेपाल','काठमाडौं','पोखरा','जनकपुर','भक्तपुर',
    'ललितपुर','बिराटनगर','धरान','इटहरी','बुटवल',
    'नारायणगढ','हेटौडा','दमक','इलाम','राजधानी',
    'नमस्ते','धन्यवाद','शुभकामना','स्वागत',
    'राम्रो','राम्री','सुन्दर','अग्लो','होचो',
    'पानी','खाना','घर','ढोका','झ्याल',
    'किताब','कलम','कापी','थैला','जुत्ता',
    'आमा','बुवा','दाई','दिदी','भाई','बहिनी',
    'साथी','शिक्षक','विद्यार्थी','विद्यालय',
    'सूर्य','चन्द्र','तारा','आकाश','पृथ्वी',
    'नदी','पहाड','जंगल','बाटो','खेत',
    'देश','समाज','सरकार','जनता','अधिकार',
    'मौसम','वर्षा','हिउँ','घाम','बादल',
    'एक','सुन्दर','हो','छ','र','को','मा',
]

def correct_ocr(text, max_distance=2):
    words       = text.split()
    corrected   = []
    corrections = []
    for word in words:
        best_word = word
        best_dist = max_distance + 1
        for dict_word in NEPALI_DICTIONARY:
            dist = edit_distance(word, dict_word)
            if dist < best_dist:
                best_dist = dist
                best_word = dict_word
        corrected.append(best_word)
        if best_word != word:
            corrections.append(f"{word} → {best_word}")
    return ' '.join(corrected), corrections

# ── Helpers ───────────────────────────────────────────────────
def ctc_decode(output):
    pred_ids = output.argmax(1).tolist()
    chars = []; prev = -1
    for idx in pred_ids:
        if idx != prev and idx != 0:
            chars.append(idx2char[idx])
        prev = idx
    return ''.join(chars)

def translate(text):
    inputs     = mt_tokenizer(text, return_tensors="pt", padding=True)
    translated = mt_model.generate(**inputs)
    return mt_tokenizer.decode(translated[0], skip_special_tokens=True)

def get_sentiment(text):
    positive_words = ['राम्रो','सुन्दर','राम्री','खुशी','आनन्द',
                      'प्रेम','मन','शुभ','सफल','उत्कृष्ट']
    negative_words = ['नराम्रो','दुःख','समस्या','गाह्रो','कठिन',
                      'रोग','मृत्यु','हार','असफल','खराब']
    words     = text.split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    if pos_count > neg_count:
        return "सकारात्मक (Positive) 😊"
    elif neg_count > pos_count:
        return "नकारात्मक (Negative) 😔"
    return "तटस्थ (Neutral) 😐"

def get_keywords(text):
    stop_words = ['एक','को','मा','छ','हो','र','मलाई',
                  'तपाईं','यो','त्यो','गर्न','भयो','छन्',
                  'भने','गरे','हुन्छ','पर्छ','हुन्']
    return [w for w in text.split() if w not in stop_words and len(w) > 1]

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_char', methods=['POST'])
def predict_char():
    data     = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    img      = Image.open(io.BytesIO(img_data)).convert('L').resize((32,32))
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
    font = ImageFont.truetype(FONT_PATH, size=48)
    img  = Image.new('L', (256, 64), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0,0), word, font=font)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((256-w)//2, (64-h)//2), word, font=font, fill=0)
    img_np = np.array(img).astype(np.float32) / 255.0
    tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    with torch.no_grad():
        out  = crnn_model(tensor)
        pred = ctc_decode(out[0])
    buf     = io.BytesIO()
    Image.fromarray((img_np*255).astype(np.uint8)).save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({'prediction': pred, 'image': img_b64})

@app.route('/analyze', methods=['POST'])
def analyze():
    text             = request.json['text']
    corrected, fixes = correct_ocr(text)
    translation      = translate(corrected)
    sentiment        = get_sentiment(corrected)
    keywords         = get_keywords(corrected)
    return jsonify({
        'translation': translation,
        'sentiment':   sentiment,
        'keywords':    keywords,
        'word_count':  len(text.split()),
        'corrected':   corrected,
        'fixes':       fixes
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
