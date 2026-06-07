import os
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from torchvision import models, transforms
from PIL import Image
import base64


app = Flask(__name__)

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224
DISPLAY_SIZE = (300, 300)

# --- MODEL ENGINE (Ensemble of DenseNet and ResNet) ---
def densenet_forward_fixed(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=False) 
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out

def load_models():
    # DenseNet121 Setup
    d = models.densenet121(weights=None)
    d.forward = densenet_forward_fixed.__get__(d, models.densenet.DenseNet)
    d.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(d.classifier.in_features, 2))
    
    # ResNet34 Setup
    r = models.resnet34(weights=None)
    r.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(r.fc.in_features, 2))

    try:
        d.load_state_dict(torch.load("densenet_best.pt", map_location=DEVICE))
        r.load_state_dict(torch.load("resnet_best.pt", map_location=DEVICE))
    except:
        print("Warning: Weights not found. Using initialized states.")

    return d.eval().to(DEVICE), r.eval().to(DEVICE)

densenet, resnet = load_models()

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- IMAGE PROCESSING UTILITIES ---
"""def is_likely_xray(pil_img):
    img_arr = np.array(pil_img)
    if len(img_arr.shape) == 2: return True 
    r, g, b = img_arr[:,:,0].astype(float), img_arr[:,:,1].astype(float), img_arr[:,:,2].astype(float)
    diff = (np.mean(np.abs(r - g)) + np.mean(np.abs(r - b))) / 2
    return diff < 15 # Reject high color saturation"""
def is_likely_xray(pil_img):
    # Convert to numpy for analysis
    img_arr = np.array(pil_img)

    # 1. Channel Variance Check (Grayscale Check)
    # If the image is already single-channel grayscale, it's likely an X-ray
    if len(img_arr.shape) == 2:
        return True

    if len(img_arr.shape) == 3:
        # Handle RGBA (4 channels) by converting to RGB
        if img_arr.shape[2] == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
            
        r, g, b = img_arr[:,:,0].astype(float), img_arr[:,:,1].astype(float), img_arr[:,:,2].astype(float)

        # Calculate average difference between color channels
        # In a real X-ray, R, G, and B values are almost identical
        diff_rg = np.mean(np.abs(r - g))
        diff_rb = np.mean(np.abs(r - b))

        # Threshold: If the average color difference is > 12, it's likely a color photo
        if diff_rg > 12 or diff_rb > 12:
            return False

    # 2. Histogram Analysis (Optional/Advanced)
    # Most X-rays have a lot of black/dark space around the lungs.
    # If the image is too bright (like a white background), it's rejected.
    if np.mean(img_arr) > 200: 
        return False

    return True
def get_encoded_img(img_np):
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def process_visuals(img_t, raw_pil, pred_label):
    # Grad-CAM logic
    feats, grads = [], []
    def save_feats(m, i, o): feats.append(o)
    def save_grads(m, gi, go): grads.append(go[0])
    target_layer = densenet.features.norm5
    h1 = target_layer.register_forward_hook(save_feats)
    h2 = target_layer.register_full_backward_hook(save_grads)
    
    img_t.requires_grad = True
    out = densenet(img_t)
    out[0, out.argmax(dim=1).item()].backward()
    
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = (w * feats[0]).sum(dim=1).relu().detach().cpu().numpy()[0]
    h1.remove(); h2.remove()
    
    cam = cv2.resize(cam, DISPLAY_SIZE)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    base_img = np.array(raw_pil.resize(DISPLAY_SIZE))
    
    # 2. Heatmap
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam_overlay = cv2.addWeighted(base_img, 0.6, heatmap, 0.4, 0)

    # 3. Localized ROI Bounding Box
    roi_img = base_img.copy()
    if pred_label == "PNEUMONIA":
        heatmap_uint8 = (cam * 255).astype(np.uint8)
        _, thresh = cv2.threshold(heatmap_uint8, int(255 * 0.6), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(roi_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return get_encoded_img(base_img), get_encoded_img(cam_overlay), get_encoded_img(roi_img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file'})
    
    img = Image.open(file.stream).convert("RGB")
    if not is_likely_xray(img): return jsonify({'error': 'Invalid Image Type'})

    img_t = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_d, out_r = torch.softmax(densenet(img_t), 1), torch.softmax(resnet(img_t), 1)
        final = (out_d + out_r) / 2
        conf, idx = final.max(1)
        pred = CLASSES[idx]

    img1, img2, img3 = process_visuals(img_t, img, pred)
    
    reasoning = "You Are Not Safe – Consult a Doctor A 'Pneumonia' finding means the AI has detected areas in your lungs that look 'cloudy' or dense, which usually happens when the air sacs are filled with fluid or infection instead of air. The AI highlighted specific 'hotspots' where it sees these patterns of illness. Because these cloudy areas can make it hard for your body to get enough oxygen, this is considered a serious finding. You should take this image to a healthcare professional immediately for a full check-up, as you may require medication or further medical treatment to clear the infection." if pred == "PNEUMONIA" else "You Are Safe A 'Normal' finding means the AI did not find any signs of infection or fluid in your lungs. In a healthy scan, the lungs appear clear because they are filled with air, which allows the X-rays to pass through easily. The AI looked at both sides of your chest and saw that the lung tissue looks healthy, open, and free of any unusual shadows or blockages. Since the system found no 'hotspots' or suspicious areas, it indicates that your breathing passages are clear and there is no evidence of illness in this scan."

    return jsonify({
        'prediction': pred,
        'confidence': f"{conf.item()*100:.1f}%",
        'reasoning': reasoning,
        'images': [img1, img2, img3]
    })

if __name__ == '__main__':
    app.run(debug=True)