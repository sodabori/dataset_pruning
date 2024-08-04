from torch.utils.data import Dataset



def get_indexed_dataset(dataset, *args, **kwargs):

    # make the subclass for dataset
    base_class = dataset.__class__

    def new_getitem(self, index):
        data, label = super(IndexedDataset, self).__getitem__(index)
        return index, data, label

    new_class_name = "IndexedDataset"
    IndexedDataset = type(
        new_class_name,
        (base_class, ),
        {'__getitem__': new_getitem})

    new_dataset = IndexedDataset(root=kwargs['root'])

    new_dataset.__dict__.update(dataset.__dict__)

    return new_dataset


def get_reinforced_dataset(dataset, *args, **kwargs):
    '''
    TODO: 다음의 기능을 수행해야함
    1. get_indexed_dataset의 기능을 수행함
    2. special index에 접근하는 경우 speical transform을 수행하도록 함.
    '''
    # make the subclass for dataset
    base_class = dataset.__class__

    def new_getitem(self, index):
        if index in self.special_indices:
            self.transform = self.special_transform
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

    return new_dataset