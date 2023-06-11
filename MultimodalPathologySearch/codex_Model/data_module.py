import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
import numpy as np
import argparse
from os import makedirs, path
from argparse import ArgumentParser
from util import calculating_stat
from dataset import codexdataset
from Utils.aux import create_dir, save_transformation, load_transformation
from pytorch_lightning.trainer.states import TrainerFn



class codex_DataModule(pl.LightningDataModule):

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        

        # -> Datasset Args

        parser.add_argument(
            "--root",
            type = str,
            default = "/home/data/nolan_lab/Tonsil_hyperstacks/bestFocus",
            help = "Address of the dataset directory."
        )
        
        
        
        parser.add_argument(
            "--transformations_write_dir",
            type = str,
            default = None,
            help = "Directory defining where to save the generated transformations and inverse transformations .obj files. If not provided, all generated coordinate files will be stored in './logs/tb_logs/logging_name/'. [default: None]"
        )

        

        parser.add_argument(
            "--patch_size",
            type = int,
            default = 64,
            help = "Size of the square patches sampled from each image. [default: 64]"
        )

        parser.add_argument(
            "--num_patches_per_image",
            type = int,
            default = 10,
            help = "Number of patches that will be sampled from each image. [default: 10]"
        )

        parser.add_argument(
            "--patching_seed",
            type = int,
            default = 4,
            help = "Seed used to generate random patches. pl.seed_everything() will not set the seed for pathcing. It should be passed manually. [default: None]"
        )


        parser.add_argument(
            "--selected_channel",
            type = int,
            default = 0,
            help = "choosing the specific channel of image. [default:0]"
        )

        parser.add_argument(
            "--whitespace_threshold",
            type = float,
            default = 0.82,
            help = "The threshold used for classifying a patch as mostly white space. The mean of pixel values over all channels of a patch after applying transformations is compared to this threshold. [default: 0.82]"
        )

        
        parser.add_argument(
            "--test_ratio",
            type = float,
            default = 0.1,
            help = ""
        )
        
        
        parser.add_argument(
            "--val_ratio",
            type = float,
            default = 0.1,
            help = ""
        )
        
        
        parser.add_argument(
            "--split_seed",
            type = int,
            default = 2,
            help = ""
        )
        
        parser.add_argument(
            "--shuffling_seed",
            type = int,
            default = 2,
            help = ""
        )


        parser.add_argument(
            "--per_image_normalize",
            action = argparse.BooleanOptionalAction,
            help = "Whether to normalize each patch with respect to itself."
        )


        parser.add_argument(
            "--prepare",
            action = argparse.BooleanOptionalAction,
            help = "getting coords."
        )

        
        

        # -> Data Module Args

        parser.add_argument(
            "--batch_size",
            type = int,
            default = 128,
            help = "The batch size used with all dataloaders. [default: 128]"
        )

        parser.add_argument(
            "--num_dataloader_workers",
            type = int,
            default = 8,
            help = "Number of processor workers used for dataloaders. [default: 8]"
        )

    
        parser.add_argument(
            "--normalize_transform",
            action = argparse.BooleanOptionalAction,
            help = "If passed, DataModule will calculate or load the whole training dataset mean and std per channel and passes it to transforms."
        )

        parser.add_argument(
            "--resize_transform_size",
            type = int,
            default = None,
            help = "If provided, the every patch would be resized from patch_size to resize_transform_size. [default: None]"
        )


        parser.add_argument(
            "--coords_read_dir",
            type = str,
            default = None,
            help = "Directory defining where to read the coords'. [default: None]"
        )

    

        parser.add_argument(
            "--coords_write_dir",
            type = str,
            default = None,
            help = "Directory defining where to write the coords'. [default: None]"
        )


        parser.add_argument(
            "--transformations_read_dir",
            type = str,
            default = None,
            help = "Directory defining where to write the coords'. [default: None]"
        )



        return parser

    def __init__(
        self,
        root,
        transformations_write_dir,
        selected_channel,
        #logging_name,
        patch_size,
        num_patches_per_image,
        patching_seed,
        whitespace_threshold,
        batch_size,
        num_dataloader_workers,
        per_image_normalize,
        normalize_transform,
        resize_transform_size,
        test_ratio,
        val_ratio,
        split_seed,
        shuffling_seed,
        coords_read_dir,
        coords_write_dir,
        prepare,
        transformations_read_dir,
        *args,
        **kwargs,
    ):
       

        super().__init__()
        
        self.root=root
        self.transformations_write_dir=transformations_write_dir
        #self.logging_name = logging_name
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.selected_channel=selected_channel
        self.patching_seed = patching_seed
        self.whitespace_threshold = whitespace_threshold
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.per_image_normalize = per_image_normalize
        self.normalize_transform = normalize_transform
        self.resize_transform_size = resize_transform_size
        self.test_ratio=test_ratio
        self.val_ratio=val_ratio
        self.split_seed=split_seed
        self.shuffling_seed=shuffling_seed
        self.coords_read_dir=coords_read_dir
        self.coords_write_dir=coords_write_dir
        self.prepare=prepare
        self.transformations_read_dir=transformations_read_dir

        # saving hyperparameters to checkpoint
        self.save_hyperparameters()

        self.dataset_kwargs = {

            "root":self.root,
            "test_ratio":self.test_ratio,
            "val_ratio":self.val_ratio,
            "split_seed":self.split_seed,
            "shuffling_seed":shuffling_seed,
            "selected_channel":selected_channel,
            "patch_size": self.patch_size,
            "num_patches_per_image": self.num_patches_per_image,
            "patching_seed": self.patching_seed,
            "whitespace_threshold": self.whitespace_threshold,
            "per_image_normalize": per_image_normalize,
            "coords_read_dir":coords_read_dir,
            "coords_write_dir":coords_write_dir
            
        }


        

        
        
        
    def prepare_data(self):
        
        if self.prepare:
           
           train_dataset = codexdataset (prepare=self.prepare, dataset_type="train",transformations=None, **self.dataset_kwargs)
                
        # All stats should be calculated at highest stable batch_size to reduce approximation errors for mean and std

           loader = DataLoader(train_dataset, batch_size=256,num_workers=self.num_dataloader_workers)
           calculating_stat(loader)

        #print((train_dataset.patches))


        #if self.trainer.state.fn == TrainerFn.PREDICTING:
            # Calculating the coordinats
            #dataset = ICdataset(dataset_type='predict', prepare=True, transformations=None, **self.dataset_kwargs)

               
                
    def setup(self, stage=None):
        

        if self.prepare:
        # Determining transformations to apply.

        
              transforms_list = []
              inverse_transforms_list = []
              final_size = self.patch_size

              if self.normalize_transform:
            
                  std = np.loadtxt(path.join("/home/axh5735/projects/compressed_images_HandE/code/with_umap/H&E_Codex/codex/logs/codex_Tonsil_30_128_256/stat", "std.gz"))
                  mean = np.loadtxt(path.join("/home/axh5735/projects/compressed_images_HandE/code/with_umap/H&E_Codex/codex/logs/codex_Tonsil_30_128_256/stat", "mean.gz"))

            

                  transforms_list.append(
                      transforms.Normalize(mean=mean, std=std)
                  )

                  #inverse_transforms_list.insert(0, transforms.Normalize(mean=-mean, std=np.array([1, 1, 1])))
                  #inverse_transforms_list.insert(0, transforms.Normalize(mean=np.array([0, 0, 0]), std=1/std))

                  inverse_transforms_list.insert(0, transforms.Normalize(mean=-mean, std=np.array([1])))
                  inverse_transforms_list.insert(0, transforms.Normalize(mean=np.array([0]), std=1/std))

              if self.resize_transform_size is not None:
                  transforms_list.append(
                      transforms.Resize(size=self.resize_transform_size, interpolation=InterpolationMode.BILINEAR)
                  )

                  inverse_transforms_list.insert(0, transforms.Resize(size=self.patch_size, interpolation=InterpolationMode.BILINEAR))

                  final_size = self.resize_transform_size

              transforms_list.append(
                  transforms.CenterCrop(final_size)
              )

              transformations = transforms.Compose(transforms_list)
              inverse_transformations = transforms.Compose(inverse_transforms_list)

              # Saving transformations to file
              save_transformation(transformations, path.join(self.transformations_write_dir, "trans.obj"))
              save_transformation(inverse_transformations, path.join(self.transformations_write_dir, "inv_trans.obj"))


        # Creating corresponding datasets

        transformations= load_transformation(path.join(self.transformations_read_dir, "trans.obj"))
        if stage in (None, "fit"):
            self.train_dataset = codexdataset(dataset_type="train", transformations=transformations
                                           ,prepare=False, **self.dataset_kwargs)
          
            self.val_dataset = codexdataset(dataset_type="val", transformations=transformations,
                                          prepare=False, **self.dataset_kwargs)
           

        elif stage in (None, "validate"):
            self.val_dataset = codexdataset(dataset_type="val", transformations=transformations
                                         ,prepare=False, **self.dataset_kwargs)
           

        elif stage in (None, "test"):
           
            self.test_dataset = codexdataset(dataset_type="test", transformations=transformations
                                          ,prepare=False, **self.dataset_kwargs)

       
        
    def train_dataloader(self):

        
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, 
        num_workers=self.num_dataloader_workers)
    
        
    
        

    
    def val_dataloader(self):
        
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
        num_workers=self.num_dataloader_workers)

    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
        num_workers=self.num_dataloader_workers)    

            
        







