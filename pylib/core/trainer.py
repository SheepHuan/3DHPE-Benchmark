import torch

class Trainer():
    
    
    def __init__(self,cfg) -> None:
        pass
    
    
    def get_optimizer(self, model):
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        return optimizer
    
    def get_dataloader(self):
        pass
    
    
    def get_model(self):
        pass
    
    
    def _train(self,max_epoch=100):
        pass