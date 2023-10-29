from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from datasets import load_dataset
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import colorama
import torch
import time
import sys
import os

DEBUG = True

def debug(data, color):
    if DEBUG:
        if not color: color = 'white'

        #region Defining Colors
        if color == 'white' or color == 0:
            color = 'WHITE'
        elif color == 'red' or color == 1:
            color = 'RED'
        elif color == 'green' or color == 2:
            color = 'GREEN'
        elif color == 'yellow' or color == 3:
            color = 'YELLOW'
        elif color == 'blue' or color == 4:
            color = 'BLUE'
        elif color == 'magenta' or color == 5:
            color = 'MAGENTA'
        elif color == 'cyan' or color == 6:
            color = 'CYAN'
        elif color == 'black' or color == 7:
            color = 'BLACK'
        elif color == 'reset' or color == 8:
            color = 'RESET'
        else:
            print(f"Wrong color! {color} is not defined! Choose from 'white', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'black', 'reset'")
            return
        #endregion

        print('\n' + getattr(colorama.Fore, color.upper()) + str(data) + colorama.Fore.RESET + '\n')



def train(arch):
    debug("Using device: " + str(device).upper(), 'yellow')
    if arch == 'mps':
        torch.multiprocessing.set_start_method('spawn')
        debug("Multiprocessing has started.\nStart method: " + str(torch.multiprocessing.get_start_method()), 'yellow')

        
        
    elif arch == 'cuda':
        torch.multiprocessing.set_start_method('spawn')

    elif arch == 'cpu':
        torch.multiprocessing.set_start_method('spawn')
    
    else:
        print("Wrong architecture! Device f{arch} not found! Please choose from 'mps', 'cuda', 'cpu'") 
        return

if torch.cuda.is_available():
    device = torch.device("cuda")
    train('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    train('mps')
elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cpu")
    train('cpu')
else:
    print("\nCUDA nor MPS not available. Using CPU\n")
    device = torch.device("cpu")
    train('cpu')