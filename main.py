import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from modules.data_loader import load_segmented_data
from modules.feature_extractors import MultiModalFeatureExtractor
from modules.model_cmt import CrossModalTransformer

def robust_parse_transcript(base_dir, vid_id):
    """Parses actual segment text to provide variance for BERT embeddings."""
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

def plot_binary_visuals(y_true, y_pred, y_probs, attn_list, title):
    """Generates visualizations with explicitly labeled axes."""
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted Sentiment Class"); plt.ylabel("True Sentiment Class")
    plt.show()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f"ROC Curve: {title}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.show()

    # 3. Cross-Modal Attention Bar Chart
    avg_attn = np.mean(np.vstack(attn_list), axis=0)
    plt.figure(figsize=(6,4))
    sns.barplot(x=['Text', 'Audio', 'Visual'], y=avg_attn, hue=['Text', 'Audio', 'Visual'], palette='viridis', legend=False)
    plt.title(f"Modality Attention Importance: {title}")
    plt.xlabel("Input Modality Type"); plt.ylabel("Attention Weight (Relative Importance)")
    plt.show()

def train():
    print("--- Starting Binary Pipeline with Full Per-Epoch Metrics ---")
    DATA_PATH = r"C:\Aparna\BITS_MTech_AIML\Sem4\mosi_memocmt\data"
    labels_df = pd.read_csv(os.path.join(DATA_PATH, "labels.csv")).set_index('segment_id')
    train_ids, test_ids = load_segmented_data(DATA_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalTransformer().to(device)
    extractor = MultiModalFeatureExtractor()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)

    for epoch in range(5):
        model.train()
        tr_losses, tr_y, tr_p = [], [], []
        print(f"\n--- Epoch {epoch+1} ---")
        
        for vid_id in train_ids[:150]:
            if vid_id not in labels_df.index: continue
            txt, found = robust_parse_transcript(DATA_PATH, vid_id)
            audio_p = os.path.join(DATA_PATH, "Raw/Audio/WAV_16000/Segmented", f"{vid_id}.wav")
            video_p = os.path.join(DATA_PATH, "Raw/Video/Segmented", f"{vid_id}.mp4")
            
            if not (os.path.exists(audio_p) and os.path.exists(video_p) and found): continue

            try:
                # Real-time extraction with variance
                t_f = extractor.get_text_features(txt).to(device)
                a_f = extractor.get_audio_features(audio_p).to(device)
                v_f = extractor.get_visual_features(video_p).to(device)
                fused = torch.stack([t_f, a_f, v_f], dim=1)
                
                optimizer.zero_grad()
                logits, _ = model(fused)
                target_score = labels_df.loc[vid_id, 'sentiment']
                target_bin = torch.tensor([[1.0 if target_score > 0 else 0.0]]).to(device)
                
                loss = criterion(logits, target_bin); loss.backward(); optimizer.step()
                tr_losses.append(loss.item()); tr_y.append(int(target_bin.item()))
                tr_p.append(1 if torch.sigmoid(logits).item() > 0.5 else 0)
            except Exception: continue
        
        # Validation Loop Per Epoch
        model.eval()
        va_losses, va_y, va_p = [], [], []
        with torch.no_grad():
            for vid_id in test_ids[:40]:
                if vid_id not in labels_df.index: continue
                txt_v, found_v = robust_parse_transcript(DATA_PATH, vid_id)
                # (Extraction logic same as train)
                try:
                    f_v = torch.stack([extractor.get_text_features(txt_v).to(device), 
                                       extractor.get_audio_features(audio_p).to(device), 
                                       extractor.get_visual_features(video_p).to(device)], dim=1)
                    v_logits, _ = model(f_v)
                    v_target = torch.tensor([[1.0 if labels_df.loc[vid_id, 'sentiment'] > 0 else 0.0]]).to(device)
                    va_losses.append(criterion(v_logits, v_target).item())
                    va_y.append(int(v_target.item())); va_p.append(1 if torch.sigmoid(v_logits).item() > 0.5 else 0)
                except Exception: continue
        
        if tr_y:
            print(f"Epoch {epoch+1} | Train Loss: {np.mean(tr_losses):.4f} | Train Acc: {accuracy_score(tr_y, tr_p)*100:.2f}%")
            print(f"Epoch {epoch+1} | Val Loss: {np.mean(va_losses):.4f} | Val Acc: {accuracy_score(va_y, va_p)*100:.2f}% | Val F1: {f1_score(va_y, va_p):.4f}")

    # --- FINAL TEST EVALUATION ---
    print("\n--- FINAL TEST LOGS (Actual Scores vs Binary Classification) ---")
    te_y, te_p, te_prob, te_attn = [], [], [], []
    with torch.no_grad():
        for vid_id in test_ids[40:60]:
            if vid_id not in labels_df.index: continue
            try:
                txt_t, _ = robust_parse_transcript(DATA_PATH, vid_id)
                f_t = torch.stack([extractor.get_text_features(txt_t).to(device), 
                                   extractor.get_audio_features(audio_p).to(device), 
                                   extractor.get_visual_features(video_p).to(device)], dim=1)
                logits, attn = model(f_t)
                prob = torch.sigmoid(logits).item()
                actual_true_score = labels_df.loc[vid_id, 'sentiment']
                
                # PRINT ACTUAL SCORES
                print(f"ID: {vid_id} | True Score: {actual_true_score:+.2f} | Pred Logit: {logits.item():+.2f} | Class: {1 if prob > 0.5 else 0} | Prob: {prob:.4f}")
                
                te_prob.append(prob); te_y.append(1 if actual_true_score > 0 else 0)
                te_p.append(1 if prob > 0.5 else 0); te_attn.append(attn.cpu().numpy())
            except Exception: continue

    if te_y: plot_binary_visuals(te_y, te_p, te_prob, te_attn, "Final Test Results")

if __name__ == "__main__":
    train()