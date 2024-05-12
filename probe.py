import numpy as np
import torch

x = torch.rand((128, 128, 3)).chunk(2)
