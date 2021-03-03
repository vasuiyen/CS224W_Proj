# Step 1 -- Install torch 1.7 with cuda 10.1
!pip install torch-geometric \
  torch-sparse==latest+cu101 \
  torch-scatter==latest+cu101 \
  torch-cluster==latest+cu101 \
  -f https://pytorch-geometric.com/whl/torch-1.7.0.html
  
# Step 2 -- Install torch scatter with torch 1.7 with cuda 10.1
!pip install torch-scatter==latest+cu101 torch-sparse==latest+cu101 -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.7.0.html
   
# Step 1 -- Install deepsnap
!pip install -q git+https://github.com/snap-stanford/deepsnap.git
  
# Now you can import torch scatter and other librarires
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn.conv import MessagePassing
##....
