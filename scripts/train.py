import torch
from torch.utils.data import DataLoader
from models.transformer import SentimentAwareTransformer
from utils.data_loader import EEGDataset
from utils.logger import get_logger

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def main():
    logger = get_logger("train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EEGDataset("data/processed/")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentimentAwareTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        logger.info(f"Epoch {epoch + 1}")
        train(model, dataloader, criterion, optimizer, device)
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch + 1}.pt")

if __name__ == "__main__":
    main()
