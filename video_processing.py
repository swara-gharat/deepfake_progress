import torch
from torchvision import models, transforms
from torch.utils.data import Dataset
from torch import nn
import face_recognition
import numpy as np
import cv2

# Dataset for video frames
class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# Model definition
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

    def get_heatmap(self, fmap, weight):
        _, nc, h, w = fmap.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        for i in range(nc):
            heatmap += weight[i] * fmap[0, i, :, :].detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / heatmap.max()
        return heatmap

# Prediction function
def predict(model, img):
    fmap, logits = model(img)
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = torch.nn.Softmax()(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    heatmap = model.get_heatmap(fmap, weight_softmax[int(prediction.item())])
    return [int(prediction.item()), confidence, heatmap]

# Function to detect fake video
def detect_fake_video(video_path):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    video_dataset = ValidationDataset([video_path], sequence_length=20, transform=transform)
    model = Model(2)
    model.load_state_dict(torch.load('df_model.pt', map_location=torch.device('cpu')))
    model.eval()
    
    prediction = predict(model, video_dataset[0])
    return prediction
