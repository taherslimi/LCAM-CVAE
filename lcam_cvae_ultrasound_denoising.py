import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

# Channel Attention Mechanism (Lightweight)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
# The rest of the code will be available after acceptance of the article 
