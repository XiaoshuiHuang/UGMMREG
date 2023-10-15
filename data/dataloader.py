import numpy
import torchvision
from data.dataset import ModelNetTest, ModelNet
import se_math.transforms as transforms


def get_categories(categoryfile):
    cinfo = None
    if categoryfile is not None:
        categories = [line.rstrip('\n') for line in open(categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
    return cinfo


# global dataset function, could call to get dataset
def get_datasets(args, dataset_type='modelnet', mode='train'):
    if dataset_type == 'modelnet':
        if mode == 'train':
            categoryfile = args.category_file
            cinfo = get_categories(categoryfile)
            transform = torchvision.transforms.Compose([
                transforms.OnUnitCube()
            ])

            traindata = ModelNet(args, train=1, transform=transform,
                                      rigid_transform=transforms.RandomTransformSE3(mag=args.mag, mag_randomly=True),
                                      classinfo=cinfo)
            valdata = ModelNet(args, train=0, transform=transform, rigid_transform=transforms.RandomTransformSE3(mag=args.mag, mag_randomly=True), classinfo=cinfo)

            return traindata, valdata
        else:
            dataset_path = args.dataset_path
            categoryfile = args.category_file
            cinfo = get_categories(categoryfile)

            # get the ground truth perturbation
            perturbations = numpy.loadtxt(args.gt_transforms, delimiter=',')

            transform = torchvision.transforms.Compose([transforms.Mesh2Points(), transforms.OnUnitCube()])

            testset = ModelNetTest(rootdir=dataset_path, transform=transform, classinfo=cinfo,
                                   perturbation=perturbations)

            return testset

    else:
        raise NotImplementedError