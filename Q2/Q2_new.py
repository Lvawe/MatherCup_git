import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建神经网络
class CustomGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(CustomGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_mlp = torch.nn.Linear(num_edge_features, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels * 2, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        edge_features = self.edge_mlp(edge_attr)

        # 结合边与点的特征
        x_combined = torch.cat([x[edge_index[0]], edge_features], dim=1)

        out = self.linear(x_combined)
        return out

# 读取数据
data_original = pd.read_csv("..\data\DC1-DC10.csv")

# 提取货物流量数据
cargo_amounts = data_original['amount'].values

# 提取节点（物流场地）
nodes = set(data_original["spot1"].unique()).union(data_original["spot2"].unique())
num_nodes = len(nodes)

# 创建节点 ID 映射，用于将节点名映射到整数 ID
node_id_map = {node: i for i, node in enumerate(nodes)}

# 提取边信息并将节点名替换为整数 ID
edges = data_original[["spot1", "spot2", "amount"]].apply(lambda x: (node_id_map[x["spot1"]], node_id_map[x["spot2"]], x["amount"]), axis=1)

X = torch.arange(0, num_nodes, dtype=torch.float).view(-1, 1)

# 边索引（源节点和目标节点的整数 ID）
edge_index_data = [(src_id, dest_id) for src_id, dest_id, *_ in edges]
edge_index = torch.tensor(edge_index_data, dtype=torch.long).t().contiguous()

# 边属性（货量）
edge_attr_data = [amount for *_, amount in edges]
edge_attr = torch.tensor(edge_attr_data, dtype=torch.float).view(-1, 1)

# 创建数据对象
data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
print(data)

# 创建模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_edge_features = data.edge_attr.size(1)
model = CustomGCN(num_node_features=data.num_node_features, num_edge_features=num_edge_features, hidden_channels=64).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 创建 target 张量
target = torch.tensor(cargo_amounts, dtype=torch.float).view(-1, 1).to(device)

# 训练模型
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, target)
    loss.backward()
    optimizer.step()

# 测试模型
model.eval()
with torch.no_grad():
    pred = model(data)

model_path = "../data/GCN_pretrained.pth"
torch.save(model.state_dict(), model_path)

# 计算均方误差（MSE）
mse = mean_squared_error(target.cpu().numpy(), pred.cpu().numpy())

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(target.cpu().numpy(), pred.cpu().numpy())

# 计算 R² 分数
r2 = r2_score(target.cpu().numpy(), pred.cpu().numpy())

print("MSE:", mse)
print("MAE:", mae)
print("R² score:", r2)

# 重新分配货物流量函数
def redistribute_cargo(data_original, dc_node,dc_neighbors):
    redistributed_data = data_original.copy()
    new_rows = []
    for index, row in data_original.iterrows():
        if row["spot1"] == dc_node:
            redistributed_amount = row["amount"] / len(dc_neighbors[row["spot2"]])
            for neighbor in dc_neighbors[row["spot2"]]:
                new_row = pd.Series(
                    {"spot1": neighbor, "spot2": row["spot2"], "date": row["date"], "amount": redistributed_amount})
                new_rows.append(new_row)
        elif row["spot2"] == dc_node:
            redistributed_amount = row["amount"] / len(dc_neighbors[row["spot1"]])
            for neighbor in dc_neighbors[row["spot1"]]:
                new_row = pd.Series(
                    {"spot1": row["spot1"], "spot2": neighbor, "date": row["date"], "amount": redistributed_amount})
                new_rows.append(new_row)
    new_rows_df = pd.DataFrame(new_rows)
    redistributed_data = pd.concat([redistributed_data, new_rows_df], ignore_index=True)
    return redistributed_data


# 输入要重新分配的 DC 节点
dc_node = input("请输入要重新分配货物的 DC 节点: ")
dc_edges = data_original[(data_original["spot1"] == dc_node) | (data_original["spot2"] == dc_node)]
dc_neighbors = {node: [n for n in nodes if n != dc_node and n != node] for node in nodes}

# 重新分配货物流量
redistributed_cargo = redistribute_cargo(data_original, dc_node,dc_neighbors)

# 加载预训练模型
model = CustomGCN(num_node_features=data.num_node_features, num_edge_features=num_edge_features, hidden_channels=64).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 使用模型预测重新分配后的货物流量
with torch.no_grad():
    pred_redistributed = model(data)

#计算变化
def count_changed_lines(daily_data_original, daily_data_redistributed):
    changed_lines_count = []
    for (date1, original_df), (date2, redistributed_df) in zip(daily_data_original, daily_data_redistributed):
        merged_df = pd.merge(original_df, redistributed_df, on=["spot1", "spot2", "date"], suffixes=("_original", "_redistributed"))
        changed_lines = 0
        for index, row in merged_df.iterrows():
            if row["amount_original"] != row["amount_redistributed"]:
                changed_lines += 1
        changed_lines_count.append((date1, changed_lines))

    return changed_lines_count

daily_data_original = list(data_original.groupby("date"))
daily_data_redistributed = list(redistributed_cargo.groupby("date"))
changed_lines_count = count_changed_lines(daily_data_original, daily_data_redistributed)

def calculate_unable_to_transfer(daily_data_original, daily_data_redistributed, max_capacity):
    unable_to_transfer = 0
    for date, (original_df_tuple, redistributed_df_tuple) in enumerate(zip(daily_data_original, daily_data_redistributed)):
        _, original_df = original_df_tuple
        _, redistributed_df = redistributed_df_tuple
        for index, row in original_df.iterrows():
            if redistributed_df.loc[index, "amount"] > max_capacity:
                unable_to_transfer += redistributed_df.loc[index, "amount"] - max_capacity

    return unable_to_transfer


max_capacity = data_original["amount"].max()
unable_to_transfer = calculate_unable_to_transfer(daily_data_original, daily_data_redistributed, max_capacity)

#标准化
def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data


def plot_redistributed_cargo(data_original, redistributed_cargo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    scaler = MinMaxScaler()
    data_original_normalized = scaler.fit_transform(data_original["amount"].values.reshape(-1, 1)).flatten()
    data_original_normalized_series = pd.Series(data_original_normalized)
    redistributed_cargo_normalized = scaler.fit_transform(redistributed_cargo["amount"].values.reshape(-1, 1)).flatten()
    redistributed_cargo_normalized_series = pd.Series(redistributed_cargo_normalized)

    ax1.scatter(data_original['spot1'], data_original['spot2'], c=data_original_normalized_series, cmap='viridis')
    ax1.set_title('Original Cargo Distribution')
    ax1.set_xlabel('Source Spot')
    ax1.set_ylabel('Destination Spot')

    ax2.scatter(redistributed_cargo['spot1'], redistributed_cargo['spot2'], c=redistributed_cargo_normalized_series, cmap='viridis')
    ax2.set_title('Redistributed Cargo Distribution')
    ax2.set_xlabel('Source Spot')
    ax2.set_ylabel('Destination Spot')

    plt.tight_layout()
    plt.show()

def plot_changed_lines(changed_lines_count):
    dates, changed_lines = zip(*changed_lines_count)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, changed_lines)
    plt.xlabel('Date')
    plt.ylabel('Changed Lines Count')
    plt.title('Number of Changed Lines per Day')
    plt.show()

def plot_unable_to_transfer(unable_to_transfer):
    plt.figure(figsize=(6, 6))
    plt.bar(['Unable to Transfer'], [unable_to_transfer])
    plt.title('Unable to Transfer Cargo Amount')
    plt.show()

# print(type(data_original))
# print("data")
# print(data_original)
# print(len(data_original['spot1']))
# data_original_normalized_series = pd.Series(data_original_normalized.flatten())
# print(len(data_original_normalized_series))
print(type(redistributed_cargo))


plot_redistributed_cargo(data_original, redistributed_cargo)
plot_changed_lines(changed_lines_count)
plot_unable_to_transfer(unable_to_transfer)