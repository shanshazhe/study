import torch
import torch.nn as nn

# Toy task: given a binary sequence, predict if sum >= T/2
T = 6
H = 32
lr = 1e-2
epochs = 100
batch_size = 64
steps_per_epoch = 50

def generate_batch(batch_size, T, device):
    # X shape: (batch_size, T), values in {0, 1}
    X = torch.randint(0, 2, (batch_size, T), device=device, dtype=torch.float32)
    # y shape: (batch_size, 1), label based on sequence sum
    y = (X.sum(dim=1) >= (T / 2)).float().unsqueeze(1)
    return X, y

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, T, input_size)
        out, _ = self.rnn(x)
        # out shape: (batch_size, T, hidden_size)
        last = out[:, -1, :]
        # logit shape: (batch_size, 1), pre-sigmoid
        logit = self.fc(last)
        return logit

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = SimpleRNN(hidden_size=H).to(device)
    # BCEWithLogitsLoss expects raw logits (no sigmoid applied)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        losses = []
        for _ in range(steps_per_epoch):
            X, y = generate_batch(batch_size, T, device)
            # Add feature dimension: (batch_size, T, 1)
            X = X.unsqueeze(-1)

            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"epoch {epoch:03d} | loss {sum(losses)/len(losses):.4f}")

    # Quick sanity check
    X_test, y_test = generate_batch(5, T, device)
    with torch.no_grad():
        preds = torch.sigmoid(model(X_test.unsqueeze(-1))).squeeze(1)
    for i in range(3):
        print(f"seq={X_test[i].int().tolist()} sum={int(X_test[i].sum())} pred={float(preds[i]):.3f} label={int(y_test[i].item())}")

if __name__ == "__main__":
    main()
