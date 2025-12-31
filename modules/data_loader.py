"""import os
import glob

def load_segmented_data(base_path):
    # Discovery of segment IDs
    video_paths = glob.glob(os.path.join(base_path, "Raw\\Video\\Segmented\\*.mp4"))
    segment_ids = [os.path.basename(f).split('.')[0] for f in video_paths]
    
    # Simple split: 70% Train, 30% Test [cite: 197]
    split_idx = int(len(segment_ids) * 0.7)
    train_ids = segment_ids[:split_idx]
    test_ids = segment_ids[split_idx:]
    
    print(f"Step 1 Verified: Loaded {len(segment_ids)} segments.")
    print(f"Train_ids segments length:{len(train_ids)}.")
    print(f"Test_ids segments length:{len(test_ids)}.")
    return train_ids, test_ids"""

import os
import glob

def load_segmented_data(base_path):
    # Search recursively (**) to find .mp4 files if they are nested
    # Also, Kaggle path is often: Raw/Video/Segmented/
    search_path = os.path.join(base_path, "Raw", "Video", "Segmented", "*.mp4")
    video_paths = glob.glob(search_path)
    
    # If standard search fails, try a recursive search
    if not video_paths:
        search_path_recursive = os.path.join(base_path, "**", "Video", "Segmented", "*.mp4")
        video_paths = glob.glob(search_path_recursive, recursive=True)

    segment_ids = [os.path.basename(f).split('.')[0] for f in video_paths]
    
    # Verification to help you debug the path
    if len(segment_ids) == 0:
        print(f"DEBUG: Searched in {search_path}, but found nothing.")
        print(f"DEBUG: Please check if this path exists: {os.path.abspath(os.path.join(base_path, 'Raw', 'Video', 'Segmented'))}")

    # Split logic
    split_idx = int(len(segment_ids) * 0.7)
    train_ids = segment_ids[:split_idx]
    test_ids = segment_ids[split_idx:]
    
    print(f"Step 1 Verified: Loaded {len(segment_ids)} segments.")
    print(f"Train_ids segments length:{len(train_ids)}.")
    print(f"Test_ids segments length:{len(test_ids)}.")
    return train_ids, test_ids