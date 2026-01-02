import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import accuracy_score, f1_score
from modules.data_loader import load_segmented_data
from modules.feature_extractors import MultiModalFeatureExtractor
from modules.model_cmt_multiclass import CrossModalTransformerMulticlass

def parse_transcript(text_path):
    """Extracts raw text from .annotprocessed files for BERT encoding."""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content.split(' ')[-1] if ' ' in content else content
    except Exception:
        return "neutral transcript"

def map_score_to_class(score):
    """Maps continuous scores [-3, 3] to class indices [0, 6] for 7-class task."""
    rounded = int(np.clip(np.round(score), -3, 3))
    return rounded + 3 

def train_multiclass():
    print("--- Starting 7-Class Training with Full Metrics & Early Stopping ---")
    
    # Configuration
    DATA_PATH = r"C:\Aparna\BITS_MTech_AIML\Sem4\mosi_memocmt\data"
    LABEL_PATH = os.path.join(DATA_PATH, "labels.csv")
    
    if not os.path.exists(LABEL_PATH):
        print(f"Error: {LABEL_PATH} not found.")
        return
        
    labels_df = pd.read_csv(LABEL_PATH).set_index('segment_id')
    train_ids, test_ids = load_segmented_data(DATA_PATH)
    
    # 1. Initialize Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalTransformerMulticlass(num_classes=7).to(device)
    extractor = MultiModalFeatureExtractor()
    
    # 2. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Early Stopping Setup
    patience = 3
    best_val_f1 = 0.0
    stop_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    # 3. Training Loop
    num_epochs = 30
    batch_limit = 100 

    for epoch in range(num_epochs):
        model.train()
        train_preds, train_labels = [], []
        epoch_loss = 0
        
        print(f"\n--- Epoch [{epoch+1}/{num_epochs}] ---")
        
        for vid_id in train_ids[:batch_limit]:
            if vid_id not in labels_df.index: continue
            
            try:
                # Triple-stream Feature Extraction
                audio_p = os.path.join(DATA_PATH, "Raw/Audio/WAV_16000/Segmented", f"{vid_id}.wav")
                video_p = os.path.join(DATA_PATH, "Raw/Video/Segmented", f"{vid_id}.mp4")
                text_p = os.path.join(DATA_PATH, "Raw/Transcript/Segmented", f"{vid_id}.annotprocessed")

                t_feat = extractor.get_text_features(parse_transcript(text_p)).to(device)
                a_feat = extractor.get_audio_features(audio_p).to(device)
                v_feat = extractor.get_visual_features(video_p).to(device)

                fused_input = torch.stack([t_feat, a_feat, v_feat], dim=1) 
                true_val = map_score_to_class(labels_df.loc[vid_id, 'sentiment'])
                target = torch.tensor([true_val]).to(device)

                # Backpropagation
                optimizer.zero_grad()
                logits = model(fused_input)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                train_preds.append(torch.argmax(logits, dim=1).item())
                train_labels.append(true_val)
            except Exception: continue

        # Calculate Training Metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        print(f"Train Loss: {epoch_loss/len(train_labels):.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.4f}")

        # --- Validation Loop (After each Epoch) ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for vid_id in test_ids[:20]: # Small validation subset for speed
                if vid_id not in labels_df.index: continue
                
                try:
                    # (Re-extracting features for validation)
                    audio_p = os.path.join(DATA_PATH, "Raw/Audio/WAV_16000/Segmented", f"{vid_id}.wav")
                    video_p = os.path.join(DATA_PATH, "Raw/Video/Segmented", f"{vid_id}.mp4")
                    text_p = os.path.join(DATA_PATH, "Raw/Transcript/Segmented", f"{vid_id}.annotprocessed")
                    
                    t_f = extractor.get_text_features(parse_transcript(text_p)).to(device)
                    a_f = extractor.get_audio_features(audio_p).to(device)
                    v_f = extractor.get_visual_features(video_p).to(device)
                    
                    out = model(torch.stack([t_f, a_f, v_f], dim=1))
                    pred = torch.argmax(out, dim=1).item()
                    real = map_score_to_class(labels_df.loc[vid_id, 'sentiment'])
                    
                    val_preds.append(pred)
                    val_labels.append(real)
                except Exception: continue

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")

        # Early Stopping Logic based on F1-Score improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            stop_counter = 0
            print("--> Validation F1 improved. Saving best model.")
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print("EARLY STOPPING TRIGGERED.")
                break

    # 4. Final Testing Loop (With True vs Pred Score logging)
    print("\n--- FINAL TEST EVALUATION ---")
    model.load_state_dict(best_model_weights)
    model.eval()
    final_preds, final_labels = [], []

    with torch.no_grad():
        for vid_id in test_ids[:30]:
            if vid_id not in labels_df.index: continue
            
            try:
                # Final Extraction for Unseen Data
                audio_p = os.path.join(DATA_PATH, "Raw/Audio/WAV_16000/Segmented", f"{vid_id}.wav")
                video_p = os.path.join(DATA_PATH, "Raw/Video/Segmented", f"{vid_id}.mp4")
                text_p = os.path.join(DATA_PATH, "Raw/Transcript/Segmented", f"{vid_id}.annotprocessed")
                
                t_f = extractor.get_text_features(parse_transcript(text_p)).to(device)
                a_f = extractor.get_audio_features(audio_p).to(device)
                v_f = extractor.get_visual_features(video_p).to(device)
                
                logits = model(torch.stack([t_f, a_f, v_f], dim=1))
                pred_class = torch.argmax(logits, dim=1).item()
                true_class = map_score_to_class(labels_df.loc[vid_id, 'sentiment'])
                
                final_preds.append(pred_class)
                final_labels.append(true_class)
                
                # Print True vs Pred Score for each segment (-3 to +3 range)
                print(f"Segment: {vid_id} | True Score: {true_class-3} | Predicted Score: {pred_class-3}")
            except Exception: continue

    # Final Metrics Summary
    final_acc = accuracy_score(final_labels, final_preds)
    final_f1 = f1_score(final_labels, final_preds, average='weighted')
    print("\n" + "="*40)
    print(f"Final Test Accuracy: {final_acc*100:.2f}%")
    print(f"Final Test F1-Score: {final_f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    train_multiclass()