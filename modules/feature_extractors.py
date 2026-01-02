import torch
import torch.nn as nn
import librosa
import cv2
from PIL import Image
from transformers import BertTokenizer, BertModel, HubertModel, Wav2Vec2Processor
from torchvision import models, transforms

class MultiModalFeatureExtractor:
    def __init__(self):
        # 1. Text: BERT-base (768D) [cite: 132, 155]
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        
        # 2. Audio: HuBERT (1024D) [cite: 130, 147]
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.audio_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        
        # 3. Visual: ResNet-50 (2048D) [cite: 131, 151]
        self.video_backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.video_backbone = nn.Sequential(*list(self.video_backbone.children())[:-1]) # Remove FC layer
        
        # 4. Projections: Map all to 256D [cite: 149, 153, 157]
        self.proj_t = nn.Linear(768, 256)
        self.proj_a = nn.Linear(1024, 256)
        self.proj_v = nn.Linear(2048, 256)

    def get_text_features(self, text_string):
        """Extracts 256D text embedding [cite: 154]"""
        inputs = self.text_tokenizer(text_string, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return self.proj_t(outputs.last_hidden_state.mean(dim=1))

    def get_audio_features(self, audio_path):
        """Extracts 256D audio embedding (prosody/tone) [cite: 146]"""
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = self.audio_processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            outputs = self.audio_model(inputs)
        return self.proj_a(outputs.last_hidden_state.mean(dim=1))

    def get_visual_features(self, video_path):
        """Extracts 256D visual embedding (facial expressions) [cite: 150]"""
        cap = cv2.VideoCapture(video_path)
        # Sample middle frame for simplicity in feature extraction 
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
        ret, frame = cap.read()
        cap.release()
        
        if not ret: return torch.zeros(1, 256)

        preprocess = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        
        with torch.no_grad():
            feat = self.video_backbone(img).flatten(1)
        return self.proj_v(feat)