import torch
import tifffile
from datetime import datetime
import random
#from Utils.aux import vips2numpy, create_dir
import numpy as np
from sklearn.model_selection import train_test_split 
#import torch
from torchvision import transforms
from os import listdir
from os.path import join
import cv2 as cv
import pickle

class codexdataset():

    def __init__(self,root,prepare,patching_seed,selected_channel,split_seed,shuffling_seed,num_patches_per_image,whitespace_threshold,patch_size,test_ratio,val_ratio
                 ,dataset_type,coords_write_dir,coords_read_dir,per_image_normalize=False,transformations=None):
         
         self.root=root
         self.patching_seed=patching_seed
         self.prepare=prepare
         self.selected_channel=selected_channel
         self.num_patches_per_image=num_patches_per_image
         self.whitespace_threshold=whitespace_threshold
         self.patch_size=patch_size
         self.per_image_normalize=per_image_normalize
         self.transformations=transformations
         self.test_ratio=test_ratio
         self.val_ratio=val_ratio
         self.split_seed=split_seed
         self.shuffling_seed=shuffling_seed
         self.dataset_type=dataset_type
         self.coords_read_dir=coords_read_dir
         self.coords_write_dir=coords_write_dir
        
        
         

            
         self.train_patches=[]
         self.val_patches=[]
         self.test_patches=[]

         if self.prepare:
            
            
            self.fnames=[]
            #i=0
            for file in listdir(self.root):
    
              if file.endswith(".tif"):
                self.fnames.append(join(self.root, file))
                #i=i+1

              #if i==2:
                #break

            print(len(self.fnames)) 
            self.train_fnames, self.test_fnames=train_test_split(self.fnames, 
                                                test_size=self.test_ratio,random_state=self.split_seed, shuffle=True)
         
            self.train_fnames, self.val_fnames = train_test_split(self.train_fnames,
                                                test_size=self.val_ratio,random_state=self.split_seed, shuffle=True)

         

            
            
            

            self._fetch_coords()


            with open(join(self.coords_write_dir,'train_coords.data'),'wb') as filehandle:
                 
                 pickle.dump(self.train_patches, filehandle)
                 filehandle.close()

            with open(join(self.coords_write_dir,'val_coords.data'),'wb') as filehandle:
                 
                 pickle.dump(self.val_patches, filehandle)
                 filehandle.close()

            with open(join(self.coords_write_dir,'test_coords.data'),'wb') as filehandle:
                 
                 pickle.dump(self.test_patches, filehandle)
                 filehandle.close()

         else:
             
            with open(join(self.coords_read_dir, 'train_coords.data' ), 'rb') as filehandle:
                self.train_patches= pickle.load(filehandle)
                filehandle.close()

            with open(join(self.coords_read_dir, 'val_coords.data' ), 'rb') as filehandle:
                self.val_patches= pickle.load(filehandle)
                filehandle.close()

            with open(join(self.coords_read_dir, 'test_coords.data' ), 'rb') as filehandle:
                self.test_patches= pickle.load(filehandle)
                filehandle.close()

            
        
         #random.shuffle(self.patches)

         #random.shuffle(patch_coords)
        
         
        



         #self.train_patches, self.test_patches=train_test_split(self.patches, 
                                                #test_size=self.test_ratio,random_state=self.split_seed, shuffle=True)
        
         #self.train_patches, self.val_patches = train_test_split(self.train_patches,
                                                #test_size=self.val_ratio,random_state=self.split_seed, shuffle=True)
         
         
    
    def _fetch_coords(self):
        
           L1=[]
           for fname in self.train_fnames:
              #print(fname, flush=True)
              patches = self._patching(fname)
              #dirs = [fname] * len(patches)
              #return list(zip(dirs, patches))
              L1.append(patches)
            
           for i in range(len(L1)):
            
              self.train_patches=L1[i]+self.train_patches

           
          

           L2=[]
           for fname in self.val_fnames:
              #print(fname, flush=True)
              patches = self._patching(fname)
              #dirs = [fname] * len(patches)
              #return list(zip(dirs, patches))
              L2.append(patches)
            
           for i in range(len(L2)):
            
              self.val_patches=L2[i]+self.val_patches

           
           
           
        
        
           L3=[]
           for fname in self.test_fnames:
              #print(fname, flush=True)
              patches = self._patching(fname)
              #dirs = [fname] * len(patches)
              #return list(zip(dirs, patches))
              L3.append(patches)
            
           for i in range(len(L3)):
            
              self.test_patches=L3[i]+self.test_patches

           random.seed(self.shuffling_seed)
           random.shuffle(self.train_patches)
           random.shuffle(self.val_patches)
           random.shuffle(self.test_patches)
           
           print(len(self.train_patches))
           print(len(self.val_patches))
           print(len(self.test_patches))
         





    def _patching(self,fname):
        
        
           random.seed(self.patching_seed)
           
            
           img=self._load_file(fname)

           

           
           
           coords=[]
           count = 0
           start_time = datetime.now()
           spent_time = datetime.now() - start_time
        
           while count < self.num_patches_per_image and spent_time.total_seconds() < 50:
               # [4, x, y] -> many [4, 512, 512]
               rand_i = random.randint(0, img.shape[0] - self.patch_size)
               rand_j = random.randint(0, img.shape[1]- self.patch_size)
            
               cropped_img=self.cropping(img,rand_i,rand_j)

               
            
        
               #output= transforms.ToTensor(cropped_img)
               output=self._img_to_tensor(cropped_img)

               

               #print((output.shape))

               
               
               
               #print(output.shape)
               if self._filter_whitespace(output, threshold=self.whitespace_threshold):
                  if self.overlap(rand_i, rand_j, coords):
                    coords.append((rand_i, rand_j,fname))
                    count += 1
                    print(count)
               spent_time = datetime.now() - start_time

               
           print('""""""""""""""""""""""final""""""""""""' + str(count)) 
           return coords


    def _img_to_tensor(self, img):
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        
        
        output= trans(img)

        #print(output.shape)
        #out_t = out_t[:3, :, :]
    
        return output
    
    def cropping(self,img,i,j):

        #print(img.shape)

        cropped_img= img[i: i+self.patch_size, j:j+self.patch_size]    
        cropped_img=cropped_img.astype(np.float32)
            
        Max=np.max(cropped_img)

        
        cropped_img=cropped_img/Max

        

        


        return cropped_img



    
        
    
    def overlap(self,i,j,coords):
    
        if len(coords) == 0:
            return True
        else:
            ml = set(map(lambda b: self.overlap_sample(b[0], b[1], i, j), coords))
            if False in ml:
                return False
            else: 
                return True
            
    def overlap_sample(self,a,b,i,j):
        
        if abs(i-a)>self.patch_size or abs(j-b)>self.patch_size :
            return True
        
        else:
            return False
        
    

    def _load_file(self, file):

        img = tifffile.imread(file)
        a,b,width,length=img.shape

        img=img.reshape(a*b,width,length)
        
        #channels=[0,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30,31,33,34,35,37,38,39,41,42,43,45,46,47,49,50,51,53,54,55,57,58,59,61,62,63,65,66,67,70,71,74,75,78,79,82,83,91]

        """
        num_channels=len(channels)

        image=[]

        for i in channels:
    
           image.append(img[:,:,i])

        image=np.array(image)

        image=image.reshape(width,length,num_channels)


        image=image[:,:,self.selected_channel]

        image= cv.merge((image,image,image))
        #print(image)
        return image
        """

        img=img[self.selected_channel,:,:]

        

        

        #out= cv.merge((img,img,img))
        #print(image)
        return img


    def _filter_whitespace(self, tensor_3d, threshold):
            
        avg= np.mean(np.array(tensor_3d[0]))
        #g = np.mean(np.array(tensor_3d[1]))
        #b = np.mean(np.array(tensor_3d[2]))
        #channel_avg = np.mean(np.array([r, g, b]))
        if avg <threshold:
            return True
        else:
            return False
        
        
    def __getitem__(self, index):
        
     if self.dataset_type=='train':
        info= self.train_patches[index]
        
     elif self.dataset_type=='val':
        info= self.val_patches[index]
        
     else:
        info= self.test_patches[index]
        
        
     tile_id = (index, index)   
        
     coord_x=info[0]
     coord_y=info[1]
     fname=  info[2]
        
     

     img = self._load_file(fname)
     patch=self.cropping(img,coord_x,coord_y)
        
     output=self._img_to_tensor(patch)
     #print(torch.min(output))
     

        

     if self.per_image_normalize:
            std, mean = torch.std_mean(output, dim=(1,2), unbiased=False)
            norm_trans = transforms.Normalize(mean=mean, std=std)
            output = norm_trans(output)

     if self.transformations is not None:
            output= self.transformations(output)

     
     if self.dataset_type in ("test", "predict"):
            return output, output.size(), fname, tile_id,coord_x,coord_y
     else:
            return output, output.size()   
     


    def __len__(self):
        
        if self.dataset_type=='train':
           return len(self.train_patches)
        
        elif self.dataset_type=='val':
        
           return len(self.val_patches)
        
        else:
            
           return len(self.test_patches)
            

