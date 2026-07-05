import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrain import train

if __name__ == "__main__":
    train()
