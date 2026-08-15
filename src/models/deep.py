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

class StreamingFusionNet(nn.Module):
    def __init__(self, per_bin_dim, flat_dim, n_classes, inner_hidden=16, outer_hidden=16, mlp_hidden=32):
        super().__init__()
        inner_out, outer_out = inner_hidden*2, outer_hidden*2
        self.inner_lstm = nn.LSTM(per_bin_dim, inner_hidden, batch_first=True, bidirectional=True)
        self.outer_lstm = nn.LSTM(inner_out, outer_hidden, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, mlp_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden, outer_out)
        )
        self.q_proj_dl = nn.Linear(outer_out, outer_out)
        self.k_proj_ml = nn.Linear(outer_out, outer_out)
        self.v_proj_ml = nn.Linear(outer_out, outer_out)
        self.q_proj_ml = nn.Linear(outer_out, outer_out)
        self.k_proj_dl = nn.Linear(outer_out, outer_out)
        self.v_proj_dl = nn.Linear(outer_out, outer_out)
        self.scale = outer_out ** 0.5
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(outer_out*2, n_classes)
        
    def forward(self, x_bins, x_flat):
        b, nw, sb, pd = x_bins.shape
        xf_seq = x_bins.view(b*nw, sb, pd)
        _, (h_n, _) = self.inner_lstm(xf_seq)
        ws = torch.cat([h_n[-2], h_n[-1]], dim=1).view(b, nw, -1)
        dl_out, _ = self.outer_lstm(ws)
        dl_out = self.dropout(dl_out)
        ml_out = self.mlp(x_flat.view(b*nw, -1)).view(b, nw, -1)
        ml_out = self.dropout(ml_out)

        q_dl = self.q_proj_dl(dl_out)
        k_ml = self.k_proj_ml(ml_out)
        v_ml = self.v_proj_ml(ml_out)
        gate_dl = torch.sigmoid((q_dl * k_ml) / self.scale)
        dl_att = gate_dl * v_ml + dl_out

        q_ml = self.q_proj_ml(ml_out)
        k_dl = self.k_proj_dl(dl_out)
        v_dl = self.v_proj_dl(dl_out)
        gate_ml = torch.sigmoid((q_ml * k_dl) / self.scale)
        ml_att = gate_ml * v_dl + ml_out

        return self.classifier(torch.cat([dl_att, ml_att], dim=-1)).squeeze(0)

def train_streaming_fusion(train_data, l2i, n_classes, per_bin_dim, flat_dim, epochs=50, seed=0):
    torch.manual_seed(seed)
    model = StreamingFusionNet(per_bin_dim, flat_dim, n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-2)
    model.train()
    import numpy as np
    for epoch in range(epochs):
        for subj, (Xb, Xf, yw, p) in train_data.items():
            y_idx = np.array([l2i[l] for l in yw])
            Xb_t = torch.tensor(Xb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            y_t = torch.tensor(y_idx, dtype=torch.long).to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(Xb_t, Xf_t), y_t)
            loss.backward()
            opt.step()
    return model
