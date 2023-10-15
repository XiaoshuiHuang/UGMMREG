import tqdm
import numpy as np
import os
import torch
import sys
sys.path.append('.')
from se_math import mesh
from data import dataset, dataloader

SAMPLE_NUM = 40000
NUM_THREAD = 24

DATASET_ROOT = './dataset'
DATA_DIR = './dataset/ModelNet40'

fileloader = mesh.offread_uniformed_trimesh
split_file = 'data/categories/modelnet40_half1.txt'

PROC_DATA_DIR = DATA_DIR + 'TrainVal'
# pattern = 

# For multi-thread processing
class DataProcessing(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        path, _ = self.samples[index]
        s = fileloader(path, sampled_pt_num=SAMPLE_NUM)
        p = mesh2points(s)

        save_path = path.replace(DATA_DIR, PROC_DATA_DIR).replace('.off', '.npy')
        dir_path = save_path.split(save_path.split('/')[-1])[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        np.save(save_path, p)

        return p

    def __len__(self):
        return len(self.samples)
        
def mesh2points(mesh):
    mesh = mesh.clone()
    v = mesh.vertex_array
    return v.astype(np.float32)

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        if not os.path.exists(DATASET_ROOT):
            os.makedirs(DATASET_ROOT)
        www = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATASET_ROOT))
        os.system('rm %s' % (zipfile))

    if os.path.exists(PROC_DATA_DIR):
        print(f"Preprocessed dataset already exits in {PROC_DATA_DIR}.")
        exit(0)
        
    classes, class_to_idx = dataloader.get_categories(split_file)
    train_samples = dataset.glob_dataset(DATA_DIR, class_to_idx, ['train/*.off'])
    test_samples = dataset.glob_dataset(DATA_DIR, class_to_idx, ['test/*.off'])

    train_loader = torch.utils.data.DataLoader( DataProcessing(train_samples), shuffle=False, drop_last=False, num_workers=NUM_THREAD)
    test_loader = torch.utils.data.DataLoader(DataProcessing(test_samples), shuffle=False, drop_last=False, num_workers=NUM_THREAD)

    if not train_samples or not test_samples:
        raise RuntimeError

    for i, _ in enumerate(tqdm.tqdm(train_loader, desc='Processing training set')):
        pass

    for i, _ in enumerate(tqdm.tqdm(test_loader, desc='Processing test set')):
        pass