import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import DEVICE

class SimpleBiLSTM(nn.Module):
    def __init__(self, per_bin_dim, n_classes, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(per_bin_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden*2, n_classes)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(torch.cat([h_n[-2], h_n[-1]], dim=1))
        return self.classifier(h)

def train_bilstm(X_train, y_train_idx, n_classes, per_bin_dim, epochs, hidden, seed):
    torch.manual_seed(seed)
    model = SimpleBiLSTM(per_bin_dim, n_classes, hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-2)
    Xt = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y_train_idx, dtype=torch.long).to(DEVICE)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt), yt)
        loss.backward(); opt.step()
    return model

class DualBranchFusionNet(nn.Module):
    def __init__(self, per_bin_dim, flat_dim, n_classes, lstm_hidden=16, mlp_hidden=32):
        super().__init__()
        out_dim = lstm_hidden * 2
        self.lstm = nn.LSTM(per_bin_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(nn.Linear(flat_dim, mlp_hidden), nn.BatchNorm1d(mlp_hidden),
                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(mlp_hidden, out_dim))
        self.q_proj_dl = nn.Linear(out_dim, out_dim); self.k_proj_ml = nn.Linear(out_dim, out_dim)
        self.v_proj_ml = nn.Linear(out_dim, out_dim)
        self.q_proj_ml = nn.Linear(out_dim, out_dim); self.k_proj_dl = nn.Linear(out_dim, out_dim)
        self.v_proj_dl = nn.Linear(out_dim, out_dim)
        self.scale = out_dim ** 0.5
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(out_dim*2, n_classes)
        
    def forward(self, x_bins, x_flat):
        _, (h_n, _) = self.lstm(x_bins)
        dl_repr = self.dropout(torch.cat([h_n[-2], h_n[-1]], dim=1))
        ml_repr = self.dropout(self.mlp(x_flat))
        
        q_dl = self.q_proj_dl(dl_repr); k_ml = self.k_proj_ml(ml_repr); v_ml = self.v_proj_ml(ml_repr)
        gate_dl = torch.sigmoid((q_dl * k_ml) / self.scale)
        dl_att = gate_dl * v_ml + dl_repr
        
        q_ml = self.q_proj_ml(ml_repr); k_dl = self.k_proj_dl(dl_repr); v_dl = self.v_proj_dl(dl_repr)
        gate_ml = torch.sigmoid((q_ml * k_dl) / self.scale)
        ml_att = gate_ml * v_dl + ml_repr
        
        return self.classifier(torch.cat([dl_att, ml_att], dim=1))

def train_fusion(Xb_train, Xf_train, y_train_idx, n_classes, per_bin_dim, flat_dim, epochs, seed):
    torch.manual_seed(seed)
    model = DualBranchFusionNet(per_bin_dim, flat_dim, n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-2)
    Xb_t = torch.tensor(Xb_train, dtype=torch.float32).to(DEVICE)
    Xf_t = torch.tensor(Xf_train, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y_train_idx, dtype=torch.long).to(DEVICE)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xb_t, Xf_t), yt)
        loss.backward(); opt.step()
    return model
