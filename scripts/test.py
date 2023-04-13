from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from scripts.model import LanguageModel
import pandas as pd
from scripts.gda import GDA
import torch
from scripts.dataset import TextDataset
    
def test(model: LanguageModel, gda : GDA, loader: DataLoader):
    device = next(model.parameters()).device
    model.eval()
    count = 0
    for date, indices, lengths, targets in tqdm(loader):
        indices = indices[:,:lengths.max()].to(device).long()
        unc = gda.predict(indices, lengths)
        pred = model(indices, lengths).argmax(1).cpu()
        batch = pd.DataFrame({
            'date':date,
            'prediction':pred.cpu(),
            'target'  :targets,
            'LogGDA'  :unc['gda'],
            'LogGMA_1':unc['gma_1'],
            'LogGMA_2':unc['gma_2'],
            'LogGMA_3':unc['gma_3'],
            'LogGMA_4':unc['gma_4'],
            'LogGMA_5':unc['gma_5']
        })
        if count == 0:
            final = batch
        else:
            final = pd.concat([final,batch], axis = 0, ignore_index = True)
        count += 1
    return final
        
        