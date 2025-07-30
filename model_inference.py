import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import pandas as pd
import json
from time import time_ns

MODEL_NUMBER = 3
device = "cpu"

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

CANON_COLUMN_INDEX = ['Fwd IAT Tot', 'Fwd Pkt Len Min', 'Down/Up Ratio', 'Dst Port', 'Fwd IAT Std', 'Fwd Header Len', 'Fwd IAT Min', 'Flow IAT Std', 'Active Std', 'Bwd IAT Max', 'Fwd Pkt Len Mean', 'Flow IAT Mean', 'Fwd Act Data Pkts', 'Bwd Pkt Len Max', 'Flow IAT Max', 'ACK Flag Cnt', 'Bwd IAT Tot', 'Flow IAT Min', 'Bwd Pkts/b Avg', 'Fwd IAT Max', 'SYN Flag Cnt', 'Bwd Header Len', 'Fwd Seg Size Avg', 'Bwd Byts/b Avg', 'Subflow Bwd Byts', 'Bwd Pkts/s', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Init Bwd Win Byts', 'Active Min', 'Subflow Fwd Byts', 'Init Fwd Win Byts', 'Bwd Pkt Len Std', 'Fwd PSH Flags', 'Fwd Pkts/s', 'Bwd Blk Rate Avg', 'Flow Byts/s', 'CWE Flag Count', 'Active Max', 'Fwd Byts/b Avg', 'Fwd Blk Rate Avg', 'Fwd Pkts/b Avg', 'Idle Mean', 'Idle Std', 'Fwd Pkt Len Max', 'Flow Duration', 'Fwd Seg Size Min', 'Bwd IAT Min', 'TotLen Fwd Pkts', 'Flow Pkts/s', 'Active Mean', 'ECE Flag Cnt', 'Idle Min', 'Subflow Bwd Pkts', 'Bwd Pkt Len Mean', 'Tot Fwd Pkts', 'Bwd IAT Std', 'Bwd Seg Size Avg', 'Bwd URG Flags', 'Bwd Pkt Len Min', 'Tot Bwd Pkts', 'Subflow Fwd Pkts', 'Bwd IAT Mean', 'FIN Flag Cnt', 'Bwd PSH Flags', 'TotLen Bwd Pkts', 'Fwd URG Flags', 'Idle Max']
CANON_COLUMN_INDEX.sort()
CANON_COLUMN_INDEX.append('Label')
index_map = {'ack_flag_cnt': 'ACK Flag Cnt', 'active_max': 'Active Max', 'active_mean': 'Active Mean', 'active_min': 'Active Min', 'active_std': 'Active Std', 'bwd_blk_rate_avg': 'Bwd Blk Rate Avg', 'bwd_byts_b_avg': 'Bwd Byts/b Avg', 'bwd_header_len': 'Bwd Header Len', 'bwd_iat_max': 'Bwd IAT Max', 'bwd_iat_mean': 'Bwd IAT Mean', 'bwd_iat_min': 'Bwd IAT Min', 'bwd_iat_std': 'Bwd IAT Std', 'bwd_iat_tot': 'Bwd IAT Tot', 'bwd_psh_flags': 'Bwd PSH Flags', 'bwd_pkt_len_max': 'Bwd Pkt Len Max', 'bwd_pkt_len_mean': 'Bwd Pkt Len Mean', 'bwd_pkt_len_min': 'Bwd Pkt Len Min', 'bwd_pkt_len_std': 'Bwd Pkt Len Std', 'bwd_pkts_b_avg': 'Bwd Pkts/b Avg', 'bwd_pkts_s': 'Bwd Pkts/s', 'bwd_seg_size_avg': 'Bwd Seg Size Avg', 'bwd_urg_flags': 'Bwd URG Flags', 'cwr_flag_count': 'CWE Flag Count', 'down_up_ratio': 'Down/Up Ratio', 'dst_port': 'Dst Port', 'ece_flag_cnt': 'ECE Flag Cnt', 'fin_flag_cnt': 'FIN Flag Cnt', 'flow_byts_s': 'Flow Byts/s', 'flow_duration': 'Flow Duration', 'flow_iat_max': 'Flow IAT Max', 'flow_iat_mean': 'Flow IAT Mean', 'flow_iat_min': 'Flow IAT Min', 'flow_iat_std': 'Flow IAT Std', 'flow_pkts_s': 'Flow Pkts/s', 'fwd_act_data_pkts': 'Fwd Act Data Pkts', 'fwd_blk_rate_avg': 'Fwd Blk Rate Avg', 'fwd_byts_b_avg': 'Fwd Byts/b Avg', 'fwd_header_len': 'Fwd Header Len', 'fwd_iat_max': 'Fwd IAT Max', 'fwd_iat_mean': 'Fwd IAT Mean', 'fwd_iat_min': 'Fwd IAT Min', 'fwd_iat_std': 'Fwd IAT Std', 'fwd_iat_tot': 'Fwd IAT Tot', 'fwd_psh_flags': 'Fwd PSH Flags', 'fwd_pkt_len_max': 'Fwd Pkt Len Max', 'fwd_pkt_len_mean': 'Fwd Pkt Len Mean', 'fwd_pkt_len_min': 'Fwd Pkt Len Min', 'fwd_pkt_len_std': 'Fwd Pkt Len Std', 'fwd_pkts_b_avg': 'Fwd Pkts/b Avg', 'fwd_pkts_s': 'Fwd Pkts/s', 'fwd_seg_size_avg': 'Fwd Seg Size Avg', 'fwd_seg_size_min': 'Fwd Seg Size Min', 'fwd_urg_flags': 'Fwd URG Flags', 'idle_max': 'Idle Max', 'idle_mean': 'Idle Mean', 'idle_min': 'Idle Min', 'idle_std': 'Idle Std', 'init_bwd_win_byts': 'Init Bwd Win Byts', 'init_fwd_w': 'Init Fwd Win Byts', 'syn_flag_cnt': 'SYN Flag Cnt', 'subflow_bwd_pkts': 'Subflow Bwd Byts', 'subflow_fwd_byts': 'Subflow Bwd Pkts', 'subflow_fwd_pkts': 'Subflow Fwd Byts', 'tot_bwd_pkts': 'Subflow Fwd Pkts', 'tot_fwd_pkts': 'Tot Bwd Pkts', 'totlen_bwd_pkts': 'Tot Fwd Pkts', 'totlen_fwd_pkts': 'TotLen Bwd Pkts', 'urg_flag_cnt': 'TotLen Fwd Pkts'}
TRAINING_UNWANTED_COLUMNS = ['Timestamp', 'Flow ID', 'Dst IP', "Src IP"]
TRAINING_WANTED_COLUMNS = []
for col in CANON_COLUMN_INDEX:
  if col not in TRAINING_UNWANTED_COLUMNS:
    TRAINING_WANTED_COLUMNS.append(col)
TRAINING_FEATURES = TRAINING_WANTED_COLUMNS[:-1]

input_shape = len(TRAINING_FEATURES) * 2
# model = DNN(
#     input_shape,
#     [int(input_shape / 2)],
#     input_shape
# )
# scripted_model = torch.jit.script(model)
# scripted_model.load_state_dict(torch.load(f'models/model_{MODEL_NUMBER}.pth', weights_only=True))

with open("models/models.json", "r") as file:
    saved_models = json.load(file)
    model_info = saved_models[MODEL_NUMBER]
    threshold = model_info['thresh']
with open("normalized/info.json", "r") as file:
    info = json.load(file)
    print(info)
    normalization_info = pd.DataFrame(info, columns=TRAINING_WANTED_COLUMNS)
loss_fn_no_reduction = nn.MSELoss(reduction='none')

#############
# GET INPUT #
#############

data = pd.read_csv("eth0.csv")
data = data.rename(columns=index_map)
data = data.reindex(columns=CANON_COLUMN_INDEX)

start_ns = time_ns()
for col in TRAINING_FEATURES:
  if info['mean'][col] != None and info['std'][col] != None:
    data[col] = (data[col] - info['mean'][col]) / info['std'][col]

x = torch.tensor(data[TRAINING_FEATURES].values, dtype=torch.float32)

# mask and impute nans
mask = torch.isnan(x).float()
x = torch.nan_to_num(x, nan=0.0)
x = torch.cat([x, mask], dim=1)

scripted_model = scripted_model.to(device)
x = x.to(device)

with torch.no_grad():
    scripted_model.eval()
    pred = scripted_model(x)
    loss = loss_fn_no_reduction(pred, x)
    l2_dist = torch.norm(loss, p=2, dim=1)
    OUTPUT = (l2_dist > threshold).tolist()

with open("output.json", "w") as file:
  s = OUTPUT.__str__()
  s.replace('\'', '\"')
  file.write(s)

###############
# SEND OUTPUT #
###############