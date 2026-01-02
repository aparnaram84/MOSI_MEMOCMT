import pandas as pd
from mmsdk import mmdatasdk

# Load the label CSD file specifically
# Path to where you stored the label CSD from the SDK or Kaggle
label_field = mmdatasdk.mmdataset({
    'labels': r'C:\Aparna\BITS_MTech_AIML\Sem4\mosi_memocmt\data\labels\CMU_MOSI_Opinion_Labels.csd'
})

# Convert to a simple dictionary: {segment_id: score}
raw_labels = label_field['labels'].data
data_for_csv = []

for seg_id in raw_labels.keys():
    # The SDK labels are often nested; we take the first value
    score = raw_labels[seg_id]['features'][0][0]
    data_for_csv.append({'segment_id': seg_id, 'sentiment': score})

# Save it so your get_label_for_segment() function can read it
df = pd.DataFrame(data_for_csv)
df.to_csv(r'C:\Aparna\BITS_MTech_AIML\Sem4\mosi_memocmt\data\labels.csv', index=False)
print("Verification: labels.csv created with", len(df), "entries.")