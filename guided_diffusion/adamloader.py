import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils

class ADAMDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['TOF-orig', 'TOF-pre']
        else:
            self.seqtypes = ['TOF-orig', 'TOF-pre', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for dir in os.listdir(self.directory):
            tof_orig_path = os.path.join(self.directory, dir, f"{dir}_TOF-orig.nii.gz")
            if not os.path.exists(tof_orig_path):
                tof_orig_path = os.path.join(self.directory, dir, f"{dir}_TOF-orig.nii") # To handle kaggle dataset

            tof_pre_path = os.path.join(self.directory, dir, f"{dir}_TOF-pre.nii.gz")
            if not os.path.exists(tof_pre_path):
                tof_pre_path = os.path.join(self.directory, dir, f"{dir}_TOF-pre.nii") # To handle kaggle dataset

            seg_path = os.path.join(self.directory, dir, f"{dir}_aneurysms.nii.gz")
            if not os.path.exists(seg_path):
                seg_path = os.path.join(self.directory, dir, f"{dir}_aneurysms.nii") # To handle kaggle dataset

            datapoint = {
                'TOF-orig': tof_orig_path,
                'TOF-pre': tof_pre_path,
                'seg': seg_path
            }
            self.database.append(datapoint)
    
    def __len__(self):
        return len(self.database) * 140

    def __getitem__(self, x):
        out = []
        n = x // 140
        slice = x % 140
        #print(x, n, slice, len(self.database))
        filedict = self.database[n]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            o = torch.tensor(nib_img.get_fdata())[:,:,slice]
            out.append(o)
        out = torch.stack(out)
        if self.test_flag:
            image=out
            if self.transform:
                image = self.transform(image)
            return (image, image, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii")
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            label=torch.where(label > 0, 1, 0).float()
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii")