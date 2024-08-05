import numpy as np


# def get_indexed_dataset(dataset, *args, **kwargs):

#     # make the subclass for dataset
#     base_class = dataset.__class__

#     def new_getitem(self, index):
#         data, label = super(IndexedDataset, self).__getitem__(index)
#         return index, data, label

#     new_class_name = "IndexedDataset"
#     IndexedDataset = type(
#         new_class_name,
#         (base_class, ),
#         {'__getitem__': new_getitem})

#     new_dataset = IndexedDataset(root=kwargs['root'])

#     new_dataset.__dict__.update(dataset.__dict__)

#     return new_dataset


def convert_to_index_dataset(dataset, init_augment='all', **kwargs):

    # make the subclass for dataset
    base_class = dataset.__class__

    def new_getitem(self, index):
        if index in self.augment_indices:
            self.transform = self.augment_transform
        else:
            self.transform = self.default_transform
        
        data, label = super(IndexedDataset, self).__getitem__(index)
        return index, data, label

    new_class_name = "IndexedDataset"
    IndexedDataset = type(
        new_class_name,
        (base_class, ),
        {'__getitem__': new_getitem})

    new_dataset = IndexedDataset(root=kwargs['root'])

    new_dataset.__dict__.update(dataset.__dict__)

    if init_augment == 'none':
        new_dataset.augment_indices = []
    elif init_augment == 'all':
        new_dataset.augment_indices = np.arange(len(new_dataset))
    else:
        raise ValueError

    return new_dataset