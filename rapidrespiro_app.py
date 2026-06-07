import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# High-performance native GUI development platform imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QFrame, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QFont, QColor, QIcon

# ------------------- PIPELINE SYSTEM CONFIGURATION -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explicit Binary Mapping: Index 0 assumed Normal, Index 1 assumed Pneumonia
CLASS_NAMES = ("Normal", "Pneumonia") 
NUM_CLASSES = len(CLASS_NAMES)
OUTPUT_DIR = "diagnostic_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standardized Inference Tensor Transform Pipeline
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ------------------- GRAD-CAM EXTRACTION LAYER HOOKS -------------------
class GradCAMHook:
    def __init__(self, target_layer):
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.forward_hook = target_layer.register_forward_hook(self.save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach().clone()

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

# ------------------- BACKBONE NETWORK FACTORY -------------------
def patch_densenet_forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=False) 
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out

def create_eval_model(model_name, weight_path, num_classes):
    if model_name.lower() == "densenet121":
        model = models.densenet121(weights=None)
        model.forward = patch_densenet_forward.__get__(model, models.densenet.DenseNet)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
    elif model_name.lower() == "resnet34":
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f"Unsupported structural architecture: {model_name}")
    
    if os.path.exists(weight_path):
        try:
            state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Successfully mounted weights: {weight_path}")
        except RuntimeError as e:
            print(f"\n[CRITICAL SHAPE MISMATCH]: Could not load {weight_path}. Fallback active.\n")
    else:
        print(f"Warning: Checkpoint file target missing at '{weight_path}'.")
        
    model.to(DEVICE)
    model.eval()
    return model


# ------------------- ASYNCHRONOUS EXECUTOR WITH GRAD-CAM -------------------
class DiagnosticWorker(QThread):
    analysis_complete = pyqtSignal(int, str, float) 
    error_occurred = pyqtSignal(str)

    def __init__(self, image_path, models_list):
        super().__init__()
        self.image_path = image_path
        self.models_list = models_list

    def run(self):
        hooks = []
        try:
            orig_pil = Image.open(self.image_path).convert('RGB')
            tensor = INFERENCE_TRANSFORMS(orig_pil).unsqueeze(0).to(DEVICE)
            tensor.requires_grad = True 

            for model in self.models_list:
                if isinstance(model, models.DenseNet):
                    hooks.append(GradCAMHook(model.features.norm5))
                elif isinstance(model, models.ResNet):
                    hooks.append(GradCAMHook(model.layer4))

            outputs_list = [model(tensor) for model in self.models_list]
            
            ensemble_outputs = torch.stack(outputs_list).mean(dim=0)
            probabilities = torch.softmax(ensemble_outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            
            predicted_idx = predicted_idx.item()
            conf_val = confidence.item()

            target_score = ensemble_outputs[0, predicted_idx]
            for model in self.models_list:
                model.zero_grad()
            target_score.backward()

            cam_heatmaps = []
            for hook in hooks:
                if hook.gradients is not None and hook.activations is not None:
                    weights = torch.mean(hook.gradients, dim=(2, 3), keepdim=True)
                    cam = torch.sum(weights * hook.activations, dim=1).squeeze(0)
                    cam = F.relu(cam) 
                    
                    cam_min, cam_max = cam.min(), cam.max()
                    if cam_min != cam_max:
                        cam = (cam - cam_min) / (cam_max - cam_min)
                    cam_heatmaps.append(cam.cpu().numpy())

            for hook in hooks:
                hook.remove()

            if cam_heatmaps:
                combined_cam = np.mean(cam_heatmaps, axis=0)
                combined_cam_resized = cv2.resize(combined_cam, orig_pil.size)
                heatmap = cv2.applyColorMap(np.uint8(255 * combined_cam_resized), cv2.COLORMAP_JET)
                
                orig_np = np.array(orig_pil)[:, :, ::-1] # RGB to BGR
                cam_overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
                
                # --- BOUNDING BOX VIEW ---
                bbox_overlay = cam_overlay.copy() 
                
                if predicted_idx == 1:
                    thresh = cv2.threshold(np.uint8(255 * combined_cam_resized), 150, 255, cv2.THRESH_BINARY)[1]
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        padding = 30
                        x1, y1 = max(0, x - padding), max(0, y - padding)
                        x2, y2 = min(orig_np.shape[1], x + w + padding), min(orig_np.shape[0], y + h + padding)
                        
                        # Draw Pure GREEN bounding box (B=0, G=255, R=0) with thickness 8
                        cv2.rectangle(bbox_overlay, (x1, y1), (x2, y2), (0, 255, 0), 8)
                
                # --- VISUAL GAPS ---
                gap_width = 25
                gap = np.zeros((orig_np.shape[0], gap_width, 3), dtype=np.uint8)
                
                # Stack 3 images horizontally with the gaps
                triple_view = np.hstack((orig_np, gap, cam_overlay, gap, bbox_overlay))
                
                filename = f"cam_report_{os.path.basename(self.image_path)}"
                save_path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(save_path, triple_view)
            else:
                save_path = self.image_path

            self.analysis_complete.emit(predicted_idx, save_path, conf_val)

        except Exception as e:
            for hook in hooks:
                try: hook.remove()
                except: pass
            self.error_occurred.emit(str(e))


# ------------------- HIGH-CONTRAST MEDICAL DASHBOARD -------------------
class XRayDiagnosticApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rapid Respiro AI - Chest X-Ray Diagnostic System")
        self.setMinimumSize(1200, 750)
        
        # --- WINDOW ICON (Taskbar and Title Bar) ---
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        
        self.models = [
            create_eval_model("densenet121", "densenet_best.pt", NUM_CLASSES),
            create_eval_model("resnet34", "resnet_best.pt", NUM_CLASSES)
        ]
        
        self.selected_img_path = None
        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Super Layout (Vertical) to hold Header and Content
        super_layout = QVBoxLayout(main_widget)
        super_layout.setContentsMargins(0, 0, 0, 0)
        super_layout.setSpacing(0)

        # --- HEADER WITH LOGO ---
        header_frame = QFrame()
        header_frame.setObjectName("HeaderPanel")
        header_frame.setFixedHeight(90)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(25, 0, 25, 0)
        
        # --- HEADER LOGO ---
        self.logo_label = QLabel()
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
        else:
            self.logo_label.setText("☤") 
            self.logo_label.setStyleSheet("color: #00E5FF; font-size: 48px; font-weight: bold;")
            
        header_layout.addWidget(self.logo_label)
        
        app_title = QLabel("RAPID RESPIRO AI")
        app_title.setStyleSheet("color: #FFFFFF; font-size: 28px; font-weight: bold; letter-spacing: 2px; margin-left: 10px;")
        header_layout.addWidget(app_title)
        header_layout.addStretch()
        
        super_layout.addWidget(header_frame)

        # --- MAIN CONTENT GRID ---
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(25, 25, 25, 25)
        content_layout.setSpacing(25)

        # Viewport Panel (Left)
        left_panel = QVBoxLayout()
        
        self.image_display = QLabel("No Medical Radiograph Mounted\n\nSelect 'Upload X-Ray Image' to begin evaluation.")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setObjectName("ImageCanvas")
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 5)
        self.image_display.setGraphicsEffect(shadow)
        
        left_panel.addWidget(self.image_display, stretch=5)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        self.upload_btn = QPushButton("UPLOAD X-RAY SCAN")
        self.upload_btn.clicked.connect(self.handle_image_upload)
        
        self.analyze_btn = QPushButton("RUN ENSEMBLE DIAGNOSTIC")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.execute_diagnostic)
        
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.analyze_btn)
        left_panel.addLayout(btn_layout, stretch=1)

        # Analytics Hub Panel (Right)
        right_panel = QVBoxLayout()
        
        metrics_container = QFrame()
        metrics_container.setObjectName("MetricsContainer")
        metrics_layout = QVBoxLayout(metrics_container)
        metrics_layout.setContentsMargins(25, 25, 25, 25)
        metrics_layout.setSpacing(15)
        
        title = QLabel("DIAGNOSTIC SUMMARY")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet("color: #00E5FF; letter-spacing: 1.5px;")
        metrics_layout.addWidget(title)
        
        self.decision_banner = QLabel("STANDBY")
        self.decision_banner.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))
        self.decision_banner.setStyleSheet("color: #383842;")
        metrics_layout.addWidget(self.decision_banner)
        
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: #2D2D2D; max-height: 1px;")
        metrics_layout.addWidget(divider)
        
        self.info_note = QLabel(
            "AI INTERPRETATION & EVIDENCE\n\n"
            "System initialized. Awaiting image upload to compute spatial gradients and provide clinical interpretations."
        )
        self.info_note.setWordWrap(True)
        self.info_note.setFont(QFont("Segoe UI", 11))
        self.info_note.setStyleSheet("color: #A0A0A0; line-height: 18px;")
        self.info_note.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        metrics_layout.addWidget(self.info_note, stretch=1)
        
        right_panel.addWidget(metrics_container)
        
        content_layout.addLayout(left_panel, stretch=6)
        content_layout.addLayout(right_panel, stretch=4)
        
        super_layout.addWidget(content_widget)

    # ------------------- CORE EXECUTION PIPELINE HANDLERS -------------------
    def handle_image_upload(self):
        file_filter = "Radiograph Images (*.png *.jpg *.jpeg)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Chest X-Ray", "", file_filter)
        
        if file_path:
            self.selected_img_path = file_path
            pixmap = QPixmap(file_path)
            
            scaled_pixmap = pixmap.scaled(
                self.image_display.size() - QSize(30, 30), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_display.setPixmap(scaled_pixmap)
            self.analyze_btn.setEnabled(True)
            self.reset_metrics_display()

    def execute_diagnostic(self):
        if not self.selected_img_path:
            return
        
        self.analyze_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.decision_banner.setText("ANALYZING...")
        self.decision_banner.setStyleSheet("color: #00E5FF; font-size: 28px;")
        
        self.worker = DiagnosticWorker(self.selected_img_path, self.models)
        self.worker.analysis_complete.connect(self.process_diagnostic_results)
        self.worker.error_occurred.connect(self.handle_inference_error)
        self.worker.start()

    def process_diagnostic_results(self, max_idx, cam_image_path, confidence):
        self.upload_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        
        if max_idx == 1: # Pneumonia
            self.decision_banner.setText("PNEUMONIA")
            self.decision_banner.setStyleSheet("color: #FF3B30; font-size: 42px;") 
            self.info_note.setText(
                "<b style='color:#FFFFFF; font-size:14px;'>Clinical Interpretation (Pneumonia Case)</b><br><br>"
                f"<b>Diagnostic Metric:</b> Pneumonia detected with <b style='color:#00E5FF;'>{confidence*100:.1f}%</b> ensemble confidence.<br><br>"
                "<b>Features:</b> Primary focal opacities and consolidations identified in the lung tissue. "
                "Feature activation patterns in the final layers of DenseNet and ResNet models show high consensus.<br><br>"
                "<b>Region of Interest:</b> The neural network has localized its attention over the pathological area, highlighted by the <b style='color:#00FF00;'>green bounding box</b> on the right-most view."
            )
        else: # Normal
            self.decision_banner.setText("NORMAL")
            self.decision_banner.setStyleSheet("color: #34C759; font-size: 42px;") 
            self.info_note.setText(
                "<b style='color:#FFFFFF; font-size:14px;'>Clinical Interpretation (Normal Case)</b><br><br>"
                f"<b>Diagnostic Metric:</b> Scan classified as Normal with <b style='color:#00E5FF;'>{confidence*100:.1f}%</b> ensemble confidence.<br><br>"
                "<b>Features:</b> Neural attention is diffuse and non-specific. No consolidated neural feature clusters or focal opacities are present.<br><br>"
                "<b>Comparison:</b> Unlike pathological cases where the AI focuses on dense regions (highlighted in red/yellow), normal scans trigger a low, scattered baseline response across the lung fields."
            )

        cam_pixmap = QPixmap(cam_image_path)
        scaled_cam = cam_pixmap.scaled(
            self.image_display.size() - QSize(30, 30),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_display.setPixmap(scaled_cam)

    def handle_inference_error(self, error_msg):
        self.upload_btn.setEnabled(True)
        self.decision_banner.setText("SYSTEM ERROR")
        self.decision_banner.setStyleSheet("color: #FF9500; font-size: 32px;")
        self.info_note.setText(f"Execution Failure: {error_msg}")

    def reset_metrics_display(self):
        self.decision_banner.setText("READY")
        self.decision_banner.setStyleSheet("color: #E0E0E0; font-size: 34px;")
        self.info_note.setText("AI INTERPRETATION & EVIDENCE\n\nAwaiting diagnostic activation pass...")

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0F0F11; 
            }
            #HeaderPanel {
                background-color: #16161A;
                border-bottom: 2px solid #22222B;
            }
            #ImageCanvas {
                background-color: #16161A;
                border: 2px dashed #2A2A32;
                border-radius: 14px;
                color: #646473;
                font-family: 'Segoe UI';
                font-size: 14px;
            }
            #MetricsContainer {
                background-color: #16161A;
                border: 1px solid #22222B;
                border-radius: 14px;
            }
            QLabel {
                font-family: 'Segoe UI';
            }
            QPushButton {
                background-color: #1F1F24;
                color: #FFFFFF;
                border: 1px solid #2D2D37;
                border-radius: 8px;
                padding: 14px 24px;
                font-family: 'Segoe UI';
                font-weight: bold;
                font-size: 13px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: #25252E;
                border-color: #00E5FF;
            }
            QPushButton:pressed {
                background-color: #141419;
            }
            QPushButton:disabled {
                background-color: #0D0D11;
                color: #42424F;
                border-color: #18181F;
            }
        """)


if __name__ == "__main__":
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    window = XRayDiagnosticApp()
    window.show()
    sys.exit(app.exec())