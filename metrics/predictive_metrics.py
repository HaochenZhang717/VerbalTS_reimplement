import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error


# class GRUPredictor(nn.Module):
#     def __init__(self, input_dim=1, hidden_dim=64):
#         super().__init__()
#         self.gru = nn.GRU(
#             input_dim,
#             hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         # x: (B, T, 1)
#         out, _ = self.gru(x)
#         y_hat = self.fc(out)
#         return y_hat


class CNNLSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, cnn_dim=32):
        super().__init__()

        # ===== CNN（提取局部pattern）=====
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ===== LSTM（建模时序）=====
        self.lstm = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # ===== 输出层 =====
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, 1)

        # 👉 CNN需要 (B, C, T)
        x = x.permute(0, 2, 1)      # (B, 1, T)

        x = self.conv(x)            # (B, cnn_dim, T)

        # 👉 LSTM需要 (B, T, C)
        x = x.permute(0, 2, 1)      # (B, T, cnn_dim)

        out, _ = self.lstm(x)       # (B, T, hidden_dim)

        y_hat = self.fc(out)        # (B, T, 1)

        return y_hat


def predictive_score_metrics(
    ori_data,
    generated_data,
    device,
    iterations=2000,
    batch_size=128,
    lr=1e-3,
):

    # Convert to tensor
    # ori_data = torch.tensor(ori_data, dtype=torch.float32).to(device)
    # generated_data = torch.tensor(generated_data, dtype=torch.float32).to(device)

    ori_data = ori_data.to(device=device, dtype=torch.float32)
    generated_data = generated_data.to(device=device, dtype=torch.float32)

    N, T, dim = generated_data.shape  # should be (N, 128, 1)

    model = CNNLSTMPredictor(input_dim=dim, hidden_dim=64, cnn_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    # ==========================
    # Train on generated data
    # ==========================
    model.train()

    for _ in range(iterations):

        idx = torch.randperm(N)[:batch_size]
        batch = generated_data[idx]  # (B, T, 1)

        X = batch[:, :-1, :]  # (B, T-1, 1)
        Y = batch[:, 1:, :]   # (B, T-1, 1)

        pred = model(X)

        loss = loss_fn(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ======================
    # Test on real data
    # ======================
    model.eval()
    MAE_temp = 0.0

    with torch.no_grad():
        N_test = ori_data.shape[0]

        for i in range(N_test):
            seq = ori_data[i:i+1]  # (1, T, 1)

            X = seq[:, :-1, :]
            Y = seq[:, 1:, :]

            pred = model(X)

            pred = pred.squeeze(0).cpu().numpy()
            Y = Y.squeeze(0).cpu().numpy()

            MAE_temp += mean_absolute_error(Y, pred)

    predictive_score = MAE_temp / N_test

    return predictive_score




