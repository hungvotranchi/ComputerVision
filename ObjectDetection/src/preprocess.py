import torch
from torchvision.transforms import transforms

    
class CustomBatchs:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = transposed_data[1]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        return (self.inp, self.tgt)
    
def collate_wrapper(batch):
    if torch.cuda.is_available():
        return CustomBatchs(batch)
    else:
        return tuple(zip(*batch))