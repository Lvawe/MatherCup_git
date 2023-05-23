import pandas as pd
import torch
from torch_geometric.data import Data

def read_data(filename):
    return pd.read_csv(filename)

def process_data(data_original):
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
    return data, cargo_amounts, node_id_map
