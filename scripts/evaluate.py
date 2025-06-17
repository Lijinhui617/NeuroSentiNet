import torch
from torch.utils.data import DataLoader
from utils.metrics import evaluate_classification
from utils.data_loader import EEGDataset
from models.transformer import SentimentAwareTransformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentAwareTransformer().to(device)
    model.load_state_dict(torch.load("checkpoints/model_epoch_10.pt"))
    model.eval()

    dataset = EEGDataset("data/processed/", train=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    evaluate_classification(y_true, y_pred)

if __name__ == "__main__":
    main()
