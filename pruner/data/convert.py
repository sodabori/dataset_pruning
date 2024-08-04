


def convert_to_index_dataset(dataset):

    # 기존 getitem을 추출
    original_getitem = dataset.__getitem__

    def new_getitem(index):
        data, label = original_getitem(index)
        return index, data, label

    # 새로운 getitem으로 바꾸기
    dataset.__getitem__ = new_getitem