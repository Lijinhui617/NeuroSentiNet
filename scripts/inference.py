import torch
import numpy as np
from models.transformer import SentimentAwareTransformer

def predict(eeg_features_path):
    model = SentimentAwareTransformer()
    model.load_state_dict(torch.load("checkpoints/model_epoch_10.pt"))
    model.eval()

    features = np.load(eeg_features_path)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    output = model(features)
    pred = torch.argmax(output, dim=1).item()
    return pred

if __name__ == "__main__":
    test_file = "data/processed/sample.npy"
    prediction = predict(test_file)
    print(f"Predicted Emotion/Cognitive State: {prediction}")
