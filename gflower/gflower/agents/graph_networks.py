import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import torch 

# adapted from https://colab.research.google.com/drive/13POV66_XKjHq3RA1Tjy-MZOFWfUA8nk9#scrollTo=jmYt-TwqOpWB
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    






