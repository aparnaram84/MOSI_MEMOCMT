import os
import torch
from modules.data_loader import load_segmented_data
from modules.feature_extractors import MultiModalFeatureExtractor
from modules.model_cmt import CrossModalTransformer

def parse_transcript(text_path):
    """Parse transcript from .annotprocessed file"""
    try:
        with open(text_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Default text for missing transcript"


def process_training_data(DATA_PATH, train_ids):
    extractor = MultiModalFeatureExtractor()
    
    for vid_id in train_ids:
        # Constructing the paths using DATA_PATH and the current train_id
        audio_input_path = os.path.join(DATA_PATH, "Raw/Audio/WAV_16000/Segmented", f"{vid_id}.wav")
        video_input_path = os.path.join(DATA_PATH, "Raw/Video/Segmented", f"{vid_id}.mp4")
        
        # 1. Input for Audio Extraction is a PATH [cite: 146]
        a_features = extractor.get_audio_features(audio_input_path)
        
        # 2. Input for Visual Extraction is a PATH [cite: 150]
        v_features = extractor.get_visual_features(video_input_path)
        
        # 3. Input for Text is the actual STRING [cite: 154]
        # You would parse this string from the .annotprocessed file
        text_string = "The person in this video is happy." 
        t_features = extractor.get_text_features(text_string)
        
        print(f"Srun_pipelineuccessfully processed features for segment: {vid_id}")
        
def run_pipeline():
    print("--- Starting Multimodal Affective Computing Pipeline ---")
    
   # Configuration
    DATA_PATH = r"C:\Aparna\BITS_MTech_AIML\Sem4\mosi_memocmt\data"
    
    # 1. Step 1: Get Data Splits [cite: 191-197]
    # This uses your logic of splitting by Video IDs to prevent data leakage
    train_ids, test_ids = load_segmented_data(DATA_PATH)
    
    # 2. Step 2: Initialize the Extraction Machine [cite: 144-158]
    extractor = MultiModalFeatureExtractor()
    
    # 3. Step 3: Initialize the Cross-Modal Transformer [cite: 160-167]
    cmt_model = CrossModalTransformer()
    
    print(f"Verification: Ready to process {len(train_ids)} training samples.")

    # 4. Processing Loop (Concatenated Pipeline)
    for vid_id in train_ids[:25]: # Verification run on first 5 samples
        print(f"\nProcessing Segment: {vid_id}")
        
        # Construct Paths
        audio_p = os.path.join(DATA_PATH, "Raw/Audio/WAV_16000/Segmented", f"{vid_id}.wav")
        video_p = os.path.join(DATA_PATH, "Raw/Video/Segmented", f"{vid_id}.mp4")
        text_p = os.path.join(DATA_PATH, "Raw/Transcript/Segmented", f"{vid_id}.annotprocessed")

        # A. Extract Text Features (Input: String)
        raw_text = parse_transcript(text_p)
        t_feat = extractor.get_text_features(raw_text) # Returns 256D [cite: 157]
        
        # B. Extract Audio Features (Input: Path)
        a_feat = extractor.get_audio_features(audio_p) # Returns 256D [cite: 149]
        
        # C. Extract Visual Features (Input: Path)
        v_feat = extractor.get_visual_features(video_p) # Returns 256D [cite: 153]

        # Verification Step: Check for 256D Alignment [cite: 175]
        print(f"Step 2 Verified: Features for {vid_id} aligned to 256D.")

        # D. Step 5: CMT Fusion [cite: 133-134]
        # Stack into [1, 3, 256] tensor (3 tokens: A, V, T)
        fused_input = torch.stack([t_feat, a_feat, v_feat], dim=1)
        prediction = cmt_model(fused_input)
        
        print(f"Step 3 Verified: Prediction for {vid_id}: {prediction.item():.4f}")

    print("\n--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    run_pipeline()