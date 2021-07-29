import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')