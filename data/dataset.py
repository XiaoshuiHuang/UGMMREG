import numpy
import torch.utils.data
import os
import glob
import copy
import six
import numpy as np
import torch
import torch.utils.data

import se_math.se3 as se3
import se_math.so3 as so3
import se_math.mesh as mesh

"""
The following three functions are defined for getting data from specific database 
"""

# find the total class names and its corresponding index from a folder
# (see the data storage structure of modelnet40)
def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the indexes from given class names
def classes_to_cinfo(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the whole 3D point cloud paths for a given class
def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []

    # loop all the folderName (class name) to find the class in class_to_idx
    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        # check if it is the class we want
        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue
        # to find the all point cloud paths in the class folder
        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
    return samples


class ModelNet(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """

    def __init__(self, config, rigid_transform, train=0, transform=None, classinfo=None):

        self.n_points = 1024
        rootdir = config.dataset_path

        if train > 0:
            self.mode = 'train'
            pattern = 'train/*.npy'
        elif train == 0:
            self.mode = 'val'
            pattern = 'test/*.npy'
        else:
            raise ValueError
        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        # find all the class names
        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        # get all the 3D point cloud paths for the class of class_to_idx
        samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        self.transform = transform
        self.rigid_transf = rigid_transform
        if self.mode == 'val':
            val_rigid_transform = []
            for i in range(len(samples)):
                val_rigid_transform.append(rigid_transform.generate_transform())
            
            self.val_rigid_transforms = val_rigid_transform

        self.classes = classes
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        define the getitem function for Dataloader of torch
        load a 3D point cloud by using a path index
        :param index:
        :return:
        """

        path, _ = self.samples[index]
        data = torch.from_numpy(np.load(path))
        ori_len = data.shape[0]
        sel = np.random.choice(ori_len, self.n_points*2, replace=False)
        sample0 = data[sel[:self.n_points]]
        sample1 = data[sel[self.n_points:]]

        sample0 = self.transform(sample0)
        sample1 = self.transform(sample1)

        if self.mode == 'train':
            sample1 = self.rigid_transf(sample1)
            
        elif self.mode == 'val':
            sample1 = self.rigid_transf.apply_transform(sample1, self.val_rigid_transforms[index])
        gt = self.rigid_transf.igt

        return sample0, sample1, gt


class ModelNetTest(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """

    def __init__(self, rootdir, transform=None, classinfo=None, perturbation=None):

        self.fileloader = mesh.offread_uniformed_trimesh

        pattern = 'test/*.off'

        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        # find all the class names
        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        # get all the 3D point cloud paths for the class of class_to_idx
        samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        # self.fileloader = fileloader
        self.transform = transform

        self.classes = classes
        self.samples = samples

        if perturbation is not None:
            self.perturbation = numpy.array(perturbation)  # twist (len(dataset), 6)
        # self.rigid_transf = rigid_transform

    def __len__(self):
        return len(self.samples)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        # x: rotation and translation
        w = x[:, 0:3]
        q = x[:, 3:6]
        R = so3.exp(w).to(p0)  # [1, 3, 3]
        g = torch.zeros(1, 4, 4)
        g[:, 3, 3] = 1
        g[:, 0:3, 0:3] = R  # rotation
        g[:, 0:3, 3] = q  # translation
        p1 = se3.transform(g, p0)
        igt = g.squeeze(0)  # igt: p0 -> p1
        return p1, igt

    def __getitem__(self, index):
        """
        define the getitem function for Dataloader of torch
        load a 3D point cloud by using a path index
        :param index:
        :return:
        """

        twist = torch.from_numpy(numpy.array(self.perturbation[index])).contiguous().view(1, 6)
        # twist = (numpy.array(self.perturbation[index])).view(1, 6)
        path, _ = self.samples[index]
        sample0 = self.fileloader(path)
        sample1 = self.fileloader(path)

        sample0 = self.transform(sample0)
        sample1 = self.transform(sample1)

        x = twist.to(sample1)
        sample1, T_gt = self.do_transform(sample1, x)

        return sample0, sample1, T_gt
