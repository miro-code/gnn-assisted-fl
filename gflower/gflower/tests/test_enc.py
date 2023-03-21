import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import torch 
from torch_geometric.utils import train_test_split_edges
from torch_geometric.datasets import Planetoid
from gflower.agents.graph_networks import GCNEncoder
from torch_geometric.nn import GAE

dataset = Planetoid("\..", "CiteSeer", transform=T.NormalizeFeatures())
print(dataset.data)

data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)

out_channels = 300
num_features = dataset.num_features
epochs = 100

# model
model = GAE(GCNEncoder(num_features, out_channels))

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
# train_pos_edge_index = data.edge_index.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

print(train_pos_edge_index)
# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    #if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    # return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, epochs + 1):
    loss = train()

    # auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}'.format(epoch, loss))

model.eval()
z  = model.encode(x, train_pos_edge_index)
print(z.shape)



