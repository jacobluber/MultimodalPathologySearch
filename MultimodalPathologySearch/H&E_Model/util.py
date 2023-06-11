import torch
import numpy as np
from os import path

def calculating_stat(loader):
        
        tot_Std=0
        tot_Mean=0
        
        for batch in loader:

            
            # Last batches may be smaller than self.batch_size
            batch_size = batch[0].shape[0]


            
            for batch_id in range(batch_size):
                Std, Mean = torch.std_mean(batch[0][batch_id], dim=(1,2), unbiased=False)


                tot_Std  =  Std  + tot_Std
                tot_Mean =  Mean + tot_Mean
    
        tot_Std  =  tot_Std / len(loader.dataset)
        tot_Mean = tot_Mean / len(loader.dataset)
            
        
        np.savetxt(path.join('/home/axh5735/projects/compressed_images_HandE/code/with_umap/H&E_Codex/H&E/logs/HandE_gray_Tonsil_128_256/stat', "std.gz"), tot_Std.numpy())
        np.savetxt(path.join('/home/axh5735/projects/compressed_images_HandE/code/with_umap/H&E_Codex/H&E/logs/HandE_gray_Tonsil_128_256/stat', "mean.gz"), tot_Mean.numpy())

        
        
        