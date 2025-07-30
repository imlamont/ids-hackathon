import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import pandas as pd
import json
from time import time_ns

MODEL_NUMBER = 3

class DNN(nn.Module):
  def __init__(self, input_size, hidden_sizes, output_size):
    super(DNN, self).__init__()

    #layers
    self.input = nn.Linear(input_size, hidden_sizes[0])
    self.output = nn.Linear(hidden_sizes[-1], output_size)
    self.dropout = nn.Dropout(0.6)
    self.hiddens = nn.ModuleList()
    for i in range(len(hidden_sizes) - 1):
      self.hiddens.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

  def forward(self, x):
    x = F.relu(self.input(x))
    for layer in self.hiddens:
      x = self.dropout(x)
      x = F.relu(layer(x))
    x = self.dropout(x)
    x = self.output(x)
    # return torch.sigmoid(x).view(-1)
    return x

CANON_COLUMN_INDEX = ['Fwd IAT Tot', 'Fwd Pkt Len Min', 'Down/Up Ratio', 'Dst Port', 'Fwd IAT Std', 'Fwd Header Len', 'Fwd IAT Min', 'Flow IAT Std', 'Active Std', 'Bwd IAT Max', 'Fwd Pkt Len Mean', 'Pkt Size Avg', 'PSH Flag Cnt', 'Flow IAT Mean', 'Fwd Act Data Pkts', 'Bwd Pkt Len Max', 'Flow IAT Max', 'ACK Flag Cnt', 'Bwd IAT Tot', 'Flow IAT Min', 'Bwd Pkts/b Avg', 'Fwd IAT Max', 'SYN Flag Cnt', 'Bwd Header Len', 'Fwd Seg Size Avg', 'Bwd Byts/b Avg', 'Subflow Bwd Byts', 'Pkt Len Max', 'Bwd Pkts/s', 'Fwd IAT Mean', 'Pkt Len Var', 'Fwd Pkt Len Std', 'Protocol', 'Init Bwd Win Byts', 'Active Min', 'Src Port', 'RST Flag Cnt', 'Subflow Fwd Byts', 'Init Fwd Win Byts', 'Bwd Pkt Len Std', 'Fwd PSH Flags', 'Fwd Pkts/s', 'Bwd Blk Rate Avg', 'Flow Byts/s', 'CWE Flag Count', 'Pkt Len Std', 'Active Max', 'Fwd Byts/b Avg', 'Fwd Blk Rate Avg', 'URG Flag Cnt', 'Timestamp', 'Fwd Pkts/b Avg', 'Idle Mean', 'Idle Std', 'Fwd Pkt Len Max', 'Pkt Len Min', 'Flow Duration', 'Fwd Seg Size Min', 'Bwd IAT Min', 'TotLen Fwd Pkts', 'Flow Pkts/s', 'Active Mean', 'ECE Flag Cnt', 'Idle Min', 'Subflow Bwd Pkts', 'Bwd Pkt Len Mean', 'Pkt Len Mean', 'Tot Fwd Pkts', 'Bwd IAT Std', 'Bwd Seg Size Avg', 'Bwd URG Flags', 'Bwd Pkt Len Min', 'Tot Bwd Pkts', 'Subflow Fwd Pkts', 'Bwd IAT Mean', 'FIN Flag Cnt', 'Bwd PSH Flags', 'TotLen Bwd Pkts', 'Fwd URG Flags', 'Idle Max']
CANON_COLUMN_INDEX.sort()
CANON_COLUMN_INDEX.append('Label')
TRAINING_UNWANTED_COLUMNS = ['Timestamp', 'Flow ID', 'Dst IP', "Src IP"]
TRAINING_WANTED_COLUMNS = []
for col in CANON_COLUMN_INDEX:
  if col not in TRAINING_UNWANTED_COLUMNS:
    TRAINING_WANTED_COLUMNS.append(col)
TRAINING_FEATURES = TRAINING_WANTED_COLUMNS[:-1]

input_shape = len(TRAINING_FEATURES) * 2
model = DNN(
    input_shape,
    [int(input_shape / 2)],
    input_shape
)
scripted_model = torch.jit.script(model)
scripted_model.load_state_dict(torch.load(f'models/model_{MODEL_NUMBER}.pth', weights_only=True))

with open("models/models.json", "r") as file:
    saved_models = json.load(file)
    model_info = saved_models[MODEL_NUMBER]
    threshold = model_info['thresh']
with open("normalized/info.json", "r") as file:
    info = json.load(file)
    normalization_info = pd.DataFrame(info, columns=TRAINING_WANTED_COLUMNS)


INPUT = [0] * len(CANON_COLUMN_INDEX)
start_ns = time_ns()
data = pd.DataFrame([INPUT], columns=CANON_COLUMN_INDEX)
for col in TRAINING_FEATURES:
  if info['mean'][col] != None and info['std'][col] != None:
    data[col] = (data[col] - info['mean'][col]) / info['std'][col]

x = torch.tensor(data[TRAINING_FEATURES].values, dtype=torch.float32)

# mask and impute nans
mask = torch.isnan(x).float()
x = torch.nan_to_num(x, nan=0.0)
x = torch.cat([x, mask], dim=1)

scripted_model.eval()
pred = scripted_model(x)
l2_dist = torch.norm(pred, p=2, dim=1)
OUTPUT = (l2_dist > threshold).tolist()
print(f"{time_ns() - start_ns}")
print(OUTPUT)