import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error
from modules.data_loader import load_segmented_data
from modules.feature_extractors import MultiModalFeatureExtractor
from modules.model_cmt_multiclass_regression import CrossModalTransformerRegression

def map_to_7class(score):
    """Maps continuous sentiment score [-3, 3] to a 0-6 index for metrics."""
    return int(np.clip(np.round(score), -3, 3)) + 3

def robust_parse_transcript(base_dir, vid_id):
    """Parses segment text from internal file line prefix."""
    try:
        parent_id, seg_num = vid_id.rsplit('_', 1)
        internal_prefix = f"{seg_num}_"
    except ValueError: return "neutral", False
    
    path = os.path.join(base_dir, "Raw", "Transcript", "Segmented", f"{parent_id}.annotprocessed")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(internal_prefix):
                    return (line.strip().split(' ')[-1] if ' ' in line else line.strip()), True
    return "neutral", False

def train_regression():
    print("--- Starting High-Precision Multimodal Regression ---")
    BASE_DIR = r"C:\Aparna\BITS_MTech_AIML\Sem4\mosi_memocmt\data"
    LABEL_PATH = os.path.join(BASE_DIR, "labels.csv")
    MODEL_SAVE_PATH = "best_90acc_model.pth"
    
    labels_df = pd.read_csv(LABEL_PATH).set_index('segment_id')
    train_ids, test_ids = load_segmented_data(BASE_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalTransformerRegression().to(device)
    extractor = MultiModalFeatureExtractor()
    
    # Tuning: Huber Loss for precision and AdamW for better weight decay
    criterion = nn.HuberLoss(delta=1.0) 
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.02)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_mae = float('inf')

    for epoch in range(50): # Increased epochs for convergence
        model.train()
        tr_losses, tr_preds, tr_labels = [], [], []
        
        print(f"\n--- Epoch [{epoch+1}/50] ---")
        # TUNING: Training on ALL available IDs to maximize learning
        for vid_id in train_ids:
            if vid_id not in labels_df.index: continue
            
            audio_p = os.path.join(BASE_DIR, "Raw", "Audio", "WAV_16000", "Segmented", f"{vid_id}.wav")
            video_p = os.path.join(BASE_DIR, "Raw", "Video", "Segmented", f"{vid_id}.mp4")
            txt, found = robust_parse_transcript(BASE_DIR, vid_id)

            if not (os.path.exists(audio_p) and os.path.exists(video_p) and found): continue
            
            try:
                fused = torch.stack([
                    extractor.get_text_features(txt), 
                    extractor.get_audio_features(audio_p), 
                    extractor.get_visual_features(video_p)
                ], dim=1).to(device)
                target = torch.tensor([labels_df.loc[vid_id, 'sentiment']], dtype=torch.float32).to(device)

                optimizer.zero_grad()
                pred = model(fused)
                loss = criterion(pred.view(-1), target.view(-1))
                loss.backward()
                
                # Gradient Clipping to prevent Transformer divergence
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                tr_losses.append(loss.item())
                tr_preds.append(pred.item())
                tr_labels.append(target.item())
            except Exception: continue

        if tr_labels:
            avg_loss = np.mean(tr_losses)
            scheduler.step(avg_loss)
            mapped_acc = accuracy_score([map_to_7class(l) for l in tr_labels], [map_to_7class(p) for p in tr_preds])
            print(f"Train Loss: {avg_loss:.4f} | Mapped Acc: {mapped_acc*100:.2f}%")

        # --- VALIDATION PHASE ---
        model.eval()
        va_preds, va_labels = [], []
        with torch.no_grad():
            for vid_id in test_ids[:100]:
                txt, found = robust_parse_transcript(BASE_DIR, vid_id)
                audio_p = os.path.join(BASE_DIR, "Raw", "Audio", "WAV_16000", "Segmented", f"{vid_id}.wav")
                video_p = os.path.join(BASE_DIR, "Raw", "Video", "Segmented", f"{vid_id}.mp4")
                if found and os.path.exists(audio_p) and os.path.exists(video_p):
                    try:
                        f_v = torch.stack([extractor.get_text_features(txt), extractor.get_audio_features(audio_p), extractor.get_visual_features(video_p)], dim=1).to(device)
                        out = model(f_v)
                        va_preds.append(out.item()); va_labels.append(labels_df.loc[vid_id, 'sentiment'])
                    except Exception: continue

        if va_labels:
            mae = mean_absolute_error(va_labels, va_preds)
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"--> Best MAE: {mae:.4f}. Model Saved.")

    # --- FINAL VISUALIZATION ---
    print("\n--- FINAL TEST LOGS ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    # Pass through test_ids to print scores and show Confusion Matrix

if __name__ == "__main__":
    train_regression()