import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix, classification_report
import seaborn
from sklearn.metrics import accuracy_score

# 1D CNN model for 500-length sequences
class Simple1DCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv1d(1, 32, kernel_size=15, padding=7),  # Preserve length (200)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 4836 -> 2418
            
            # Second block
            nn.Conv1d(32, 64, kernel_size=7, padding=3),  # Preserve length (100)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 2418 -> 1209
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1209, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def show_menu():
    print("\n=== Main Menu ===")
    print("1. Train")
    print("2. Evaluate")
    print("3. Save")
    print("4. Load")
    print("5. Exit")

def load_data_set(filename):
    data = np.loadtxt(filename, delimiter=",")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

def train():
    xdataset, ydataset = load_data_set("numbersSoftMediumHard.csv")
    #trim datasets to an even number
    xdataset = np.array([x[:4836] for x in xdataset])

    #plot the first of each class
    plt.plot(xdataset[ydataset == 0][0], label="class 0")
    plt.plot(xdataset[ydataset == 1][0], label="class 1")
    plt.plot(xdataset[ydataset == 2][0], label="class 2")
    plt.legend(loc="best")
    plt.show()

     # Convert to PyTorch tensors (using your new X, y with length=500)
    X_tensor = torch.FloatTensor(xdataset).unsqueeze(1)  # Shape: [num_samples, 1, 500]
    y_tensor = torch.LongTensor(ydataset)

    # Split into train/test (80/20)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders with larger batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = Simple1DCNN1()

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Improved training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Training loop
    num_epochs = 40

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        train_acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')

    return model, test_loader, loss

def evaluate(model, test_loader, loss):
    total = 0
    correct = 0

    model.eval()

    for batch in test_loader:
        # Explicitly unpack data and labels (handles both tuple and tensor cases)
        datas, labels = batch  
        print(labels)
        
        # Convert labels to tensor if they aren't already
        if isinstance(labels, tuple):
            labels = labels.float().unsqueeze(1)  # Convert labels to float and reshape to (batch_size, 1)
            # labels = torch.tensor(labels[0])  # Take first element if labels is a tuple
        
        # Ensure proper type
        labels = labels.long()  # Convert to long integers
        
        outputs = model(datas)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        print(f'Loss: {loss.item():.4f} | Acc: {100*correct/total:.2f}%')
    matrix = confusion_matrix(labels,predicted,labels=np.unique(labels))
    plt.figure(figsize=(10,10))
    seaborn.heatmap(matrix,annot=True, fmt='d', xticklabels=np.unique(labels),yticklabels=np.unique(labels),cmap="Greens")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.show()

    accuracy = accuracy_score(labels, predicted)
    print(predicted)
    print(accuracy)

while True:
    show_menu()
    choice = input("Enter your choice (1-4): ")

    if choice == "1":
       model, test_loader, loss = train()
    elif choice == "2":
        evaluate(model, test_loader, loss)
    elif choice == "3":
        save()
    elif choice == "4":
        load()
    elif choice == "5":
        print("Exiting program. Goodbye!")
        break
    else:
        print("Invalid choice. Please try again.")