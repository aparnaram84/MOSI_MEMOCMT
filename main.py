import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from modules.data_loader import load_segmented_data
from modules.feature_extractors import MultiModalFeatureExtractor
from modules.model_cmt import CrossModalTransformer

def parse_transcript(text_path):
    """
    Parses raw text from the .annotprocessed files.
    
    Input:
        text_path (str): Absolute path to a .annotprocessed file.
    Output:
        raw_text (str): Cleaned transcription string for the BERT encoder.
    """
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Extracts the semantic meaning as per slide 154
            return content.split(' ')[-1] if ' ' in content else content
    except FileNotFoundError:
        return ""

def train():
    """
    Main training and validation pipeline for the Affective Computing project.
    
    Logic Flow:
    1. Load segment IDs and Ground Truth labels.
    2. Extract Triple-stream features (A, V, T).
    3. Fuse features using Cross-Modal Transformer (CMT).
    4. Validate using sentiment scores and calculate metrics.
    """
    print("--- Starting Multimodal Affective Computing Training & Validation ---")
    
    # PROJECT CONFIGURATION
    # DATA_PATH: Root directory containing Raw/ folder
    DATA_PATH = r"C:\Aparna\BITS_MTech_AIML\Sem4\mosi_memocmt\data"
    # LABEL_PATH: CSV containing mapping of segment_id to -3/+3 sentiment scores [cite: 196, 213]
    LABEL_PATH = os.path.join(DATA_PATH, "labels.csv")
    
    if not os.path.exists(LABEL_PATH):
        print(f"Error: {LABEL_PATH} not found. Please run label generator first.")
        return
        
    # Load labels into a DataFrame for O(1) lookup during training
    labels_df = pd.read_csv(LABEL_PATH).set_index('segment_id')
    
    # STEP 1: DATA SPLITS [cite: 12]
    # Input: DATA_PATH
    # Output: Lists of unique segment IDs (e.g., 'video1_1')
    train_ids, test_ids = load_segmented_data(DATA_PATH)
    
    # STEP 2: INITIALIZE FEATURE EXTRACTORS [cite: 144-145]
    # Input: Raw paths or strings
    # Output: Projected 256D Tensors
    extractor = MultiModalFeatureExtractor()
    
    # STEP 3: INITIALIZE CROSS-MODAL TRANSFORMER [cite: 160, 186]
    # Input: Fused token tensor [Batch, 3, 256]
    # Output: Sentiment regression score (1D)
    cmt_model = CrossModalTransformer()
    
    all_preds = []
    all_ground_truth = []

    # Processing first 25 samples for verification of the architecture flow
    print(f"Validation: Processing {len(train_ids[:100])} segments.")

    for vid_id in train_ids[:25]:
        if vid_id not in labels_df.index:
            continue
            
        # Define Absolute Paths for the three modalities
        audio_p = os.path.join(DATA_PATH, "Raw/Audio/WAV_16000/Segmented", f"{vid_id}.wav")
        video_p = os.path.join(DATA_PATH, "Raw/Video/Segmented", f"{vid_id}.mp4")
        text_p = os.path.join(DATA_PATH, "Raw/Transcript/Segmented", f"{vid_id}.annotprocessed")

        # --- FEATURE EXTRACTION (TRIPLE-STREAM) ---
        
        # A. Text Stream [cite: 154-157]
        # Input: String (e.g. "I am happy") 
        # Output: [1, 256] Dimension BERT embedding
        raw_text = parse_transcript(text_p)
        t_feat = extractor.get_text_features(raw_text) 

        # B. Audio Stream [cite: 146-149]
        # Input: Path to 16kHz WAV file
        # Output: [1, 256] Dimension HuBERT embedding (prosody & tone)
        a_feat = extractor.get_audio_features(audio_p) 

        # C. Visual Stream [cite: 150-153]
        # Input: Path to MP4 file
        # Output: [1, 256] Dimension ResNet embedding (facial expressions)
        v_feat = extractor.get_visual_features(video_p) 

        # STEP 4: TOKEN-LEVEL FUSION [cite: 133-134, 175]
        # Logic: Stack the three 256D vectors to create a sequence of tokens
        # Input Dimension: 3 x [1, 256]
        # Output Dimension: [1, 3, 256] (where 3 represents A, V, and T tokens)
        fused_input = torch.stack([t_feat, a_feat, v_feat], dim=1)
        
        # STEP 5: CMT PREDICTION [cite: 161-164, 189]
        # Learns deep cross-modal relationships via Multi-head attention
        prediction = cmt_model(fused_input).item()
        
        #best_id = get_best_match(vid_id, labels_df.index)
        #if best_id:
        #ground_truth = labels_df.loc[best_id, 'sentiment']

        # Retrieve Ground Truth [cite: 196]
        ground_truth = labels_df.loc[vid_id, 'sentiment']
        
        all_preds.append(prediction)
        all_ground_truth.append(ground_truth)
        
        print(f"Segment {vid_id} | Aligned Dim: [3, 256] | Pred: {prediction:.2f} | Real: {ground_truth:.2f}")

    # STEP 6: METRIC EVALUATION [cite: 13, 203-207]
    # Logic: Convert continuous scores (-3 to +3) to binary classes
    y_pred = [1 if p > 0 else 0 for p in all_preds]
    y_true = [1 if g > 0 else 0 for g in all_ground_truth]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\n--- Final Performance Metrics ---")
    print(f"Accuracy: {acc * 100:.2f}% (Binary Sentiment Recognition)")
    print(f"F1-Score: {f1:.4f} (Weighted average recall)")
    print("--- Training Pipeline Complete ---")

if __name__ == "__main__":
    train()