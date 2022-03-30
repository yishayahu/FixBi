import os
from pathlib import Path
from typing import Tuple, Any

from torchvision import datasets
import torchvision.transforms as transforms

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        imname = path.split(os.sep)[-2:]
        imname = '_'.join(imname)

        return sample, target,imname


def get_dataset(dataset_name, office_path='/database'):
    office_path = Path(office_path)
    if dataset_name in ['amazon', 'dslr', 'webcam']:  # OFFICE-31
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        tr_dataset = MyImageFolder(office_path / 'office31' / dataset_name /'images', data_transforms['train'])
        te_dataset = MyImageFolder(office_path / 'office31' / dataset_name /'images', data_transforms['test'])
        print('{} train set size: {}'.format(dataset_name, len(tr_dataset)))
        print('{} test set size: {}'.format(dataset_name, len(te_dataset)))

    else:
        raise ValueError('Dataset %s not found!' % dataset_name)
    return tr_dataset, te_dataset
