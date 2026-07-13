import re

with open("paper_experiments/train_and_compare.py", "r") as f:
    text = f.read()

# 1. Update EPOCHS and SEEDS if needed
text = re.sub(r'SEEDS\s*=\s*30', 'SEEDS = 30', text)
text = re.sub(r'EPOCHS\s*=\s*50', 'EPOCHS = 250', text)

# 2. Replace BioLiquidCellImpl fallback with the Full CfC definition directly
text = re.sub(
    r'class BioLiquidCellImpl\(nn\.Module\):.*?def forward.*?\n\s*return.*?t_gate\n',
    '',
    text,
    flags=re.DOTALL
)
# And just hardcode the ContinuousCfC as the full one:
full_cfc_code = '''
class ContinuousCfC(nn.Module):
    """Full 3-branch CfC (Hasani et al. [2022])."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, backbone_units: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, backbone_units), nn.Tanh()
        )
        self.f_head = nn.Linear(backbone_units, hidden_dim)
        self.g_head = nn.Linear(backbone_units, hidden_dim)
        self.h_head = nn.Linear(backbone_units, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._hsize = hidden_dim

    @property
    def _hsize_prop(self):
        return self._hsize

    def forward(self, x: torch.Tensor, times: torch.Tensor):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self._hsize, device=x.device)
        out = []
        for t in range(seq_len):
            dt = torch.zeros(batch, 1, device=x.device) if t == 0 else (times[:, t, :] - times[:, t-1, :])
            bb = self.backbone(torch.cat([x[:, t, :], h], dim=-1))
            t_gate = torch.sigmoid(-self.f_head(bb) * dt)
            h = t_gate * torch.tanh(self.g_head(bb)) + (1.0 - t_gate) * torch.tanh(self.h_head(bb))
            out.append(h.unsqueeze(1))
        return self.fc(torch.cat(out, dim=1))
'''

text = re.sub(
    r'class ContinuousCfC\(nn\.Module\):.*?return self\.fc\(torch\.cat\(out, dim=1\)\)\n\n\nclass ContinuousCfCFull\(nn\.Module\):.*?return self\.fc\(torch\.cat\(out, dim=1\)\)\n',
    full_cfc_code,
    text,
    flags=re.DOTALL
)

# 3. Remove CfCFull from the training loop
text = re.sub(
    r'models\["CfCFull"\]\s*=\s*ContinuousCfCFull\(4, hidden_cfc, 1\).*?models\["CfCFull"\]\s*=\s*None',
    '',
    text,
    flags=re.DOTALL
)
# Fix any remaining CfCFull dictionary initialization
text = re.sub(r',\s*"CfCFull":\s*\[\]', '', text)
text = re.sub(r',\s*"CfCFull":\s*ContinuousCfCFull\(4, hidden_cfc, 1\)', '', text)

with open("paper_experiments/train_and_compare.py", "w") as f:
    f.write(text)
