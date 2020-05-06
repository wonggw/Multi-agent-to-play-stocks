import gc
import torch
from JobSystem import Manager

if __name__ == '__main__':
    gc.collect()    
    torch.cuda.empty_cache()
    Manager.manager()