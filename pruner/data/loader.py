import torch
import numpy as np

from contextlib import suppress
from functools import partial
from typing import Callable, Optional, Tuple, Union

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.dataset import IterableImageDataset, ImageDataset
from timm.data.transforms_factory import create_transform

from timm.data.loader import RepeatAugSampler, MultiEpochsDataLoader, OrderedDistributedSampler, _worker_init
from timm.data.loader import fast_collate as fast_collate_for_input_target
from timm.data.loader import PrefetchLoader as InputTargetPrefetchLoader

from timm.data.random_erasing import RandomErasing


def fast_collate_for_index_input_target(batch):
    '''
    fast_collate function for (index, input, target) data tuple
    '''

    assert isinstance(batch[0], tuple)
    batch_size = len(batch)

    numpy_array = np.array([b[0] for b in batch])
    indices = torch.tensor(numpy_array, dtype=torch.int64)

    if isinstance(batch[0][1], np.ndarray):
        targets = torch.tensor([b[2] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][1].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][1])
        return indices, tensor, targets
    elif isinstance(batch[0][1], torch.Tensor):
        targets = torch.tensor([b[2] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][1].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][1])
        return indices, tensor, targets
    else:
        assert False


class IndexInputTargetPrefetchLoader(InputTargetPrefetchLoader):

    def __init__(
            self,
            loader,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            channels=3,
            device=torch.device('cuda'),
            img_dtype=torch.float32,
            fp16=False,
            re_prob=0.,
            re_mode='const',
            re_count=1,
            re_num_splits=0):
        
        super(IndexInputTargetPrefetchLoader, self).__init__(
            loader,
            mean=mean,
            std=std,
            channels=channels,
            device=device,
            img_dtype=img_dtype,
            fp16=fp16,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits)
        
    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_index, next_input, next_target in self.loader:

            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_target = next_target.to(device=self.device, non_blocking=True)
                next_input = next_input.to(self.img_dtype).sub_(self.mean).div_(self.std)

                # Random erasing is only used for augment indices
                if self.random_erasing is not None:
                    next_augment_input = self.random_erasing(next_input)

                    augment_index = [i for i, index in enumerate(self.loader.dataset.augment_indices) if index in set(next_index)]

                    next_input[augment_index] = next_augment_input[augment_index]

            if not first:
                yield index, input, target
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)
            
            index = next_index
            input = next_input
            target = next_target

        yield index, input, target

def create_loader(
        dataset: Union[ImageDataset, IterableImageDataset],
        input_size: Union[int, Tuple[int, int], Tuple[int, int, int]],
        batch_size: int,
        is_training: bool = False,
        no_aug: bool = False,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_split: bool = False,
        train_crop_mode: Optional[str] = None,
        scale: Optional[Tuple[float, float]] = None,
        ratio: Optional[Tuple[float, float]] = None,
        hflip: float = 0.5,
        vflip: float = 0.,
        color_jitter: float = 0.4,
        color_jitter_prob: Optional[float] = None,
        grayscale_prob: float = 0.,
        gaussian_blur_prob: float = 0.,
        auto_augment: Optional[str] = None,
        num_aug_repeats: int = 0,
        num_aug_splits: int = 0,
        interpolation: str = 'bilinear',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        num_workers: int = 1,
        distributed: bool = False,
        crop_pct: Optional[float] = None,
        crop_mode: Optional[str] = None,
        crop_border_pixels: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        fp16: bool = False,  # deprecated, use img_dtype
        img_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cuda'),
        use_prefetcher: bool = True,
        use_multi_epochs_loader: bool = False,
        persistent_workers: bool = True,
        worker_seeding: str = 'all',
        tf_preprocessing: bool = False,
        augment_indices: Optional[list] = None,
):
    """

    Args:
        dataset: The image dataset to load.
        input_size: Target input size (channels, height, width) tuple or size scalar.
        batch_size: Number of samples in a batch.
        is_training: Return training (random) transforms.
        no_aug: Disable augmentation for training (useful for debug).
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_split: Control split of random erasing across batch size.
        scale: Random resize scale range (crop area, < 1.0 => zoom in).
        ratio: Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug
        grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
        gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
        auto_augment: Auto augment configuration string (see auto_augment.py).
        num_aug_repeats: Enable special sampler to repeat same augmentation across distributed GPUs.
        num_aug_splits: Enable mode where augmentations can be split across the batch.
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        num_workers: Num worker processes per DataLoader.
        distributed: Enable dataloading for distributed training.
        crop_pct: Inference crop percentage (output size / resize size).
        crop_mode: Inference crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.
        crop_border_pixels: Inference crop border of specified # pixels around edge of original image.
        collate_fn: Override default collate_fn.
        pin_memory: Pin memory for device transfer.
        fp16: Deprecated argument for half-precision input dtype. Use img_dtype.
        img_dtype: Data type for input image.
        device: Device to transfer inputs and targets to.
        use_prefetcher: Use efficient pre-fetcher to load samples onto device.
        use_multi_epochs_loader:
        persistent_workers: Enable persistent worker processes.
        worker_seeding: Control worker random seeding at init.
        tf_preprocessing: Use TF 1.0 inference preprocessing for testing model ports.

    Returns:
        DataLoader
    """
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2

    if augment_indices is None:
        dataset.transform = create_transform(
            input_size,
            is_training=is_training,
            no_aug=no_aug,
            train_crop_mode=train_crop_mode,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            color_jitter_prob=color_jitter_prob,
            grayscale_prob=grayscale_prob,
            gaussian_blur_prob=gaussian_blur_prob,
            auto_augment=auto_augment,
            interpolation=interpolation,
            mean=mean,
            std=std,
            crop_pct=crop_pct,
            crop_mode=crop_mode,
            crop_border_pixels=crop_border_pixels,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            tf_preprocessing=tf_preprocessing,
            use_prefetcher=use_prefetcher,
            separate=num_aug_splits > 0,
        )
    else:
        dataset.default_transform = create_transform(
            input_size,
            is_training=is_training,
            no_aug=True,
            interpolation=interpolation,
            mean=mean,
            std=std,
        )

        dataset.augment_transform = create_transform(
            input_size,
            is_training=is_training,
            no_aug=no_aug,
            train_crop_mode=train_crop_mode,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            color_jitter_prob=color_jitter_prob,
            grayscale_prob=grayscale_prob,
            gaussian_blur_prob=gaussian_blur_prob,
            auto_augment=auto_augment,
            interpolation=interpolation,
            mean=mean,
            std=std,
            crop_pct=crop_pct,
            crop_mode=crop_mode,
            crop_border_pixels=crop_border_pixels,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            tf_preprocessing=tf_preprocessing,
            use_prefetcher=use_prefetcher,
            separate=num_aug_splits > 0,
        )

    if isinstance(dataset, IterableImageDataset):
        # give Iterable datasets early knowledge of num_workers so that sample estimates
        # are correct before worker processes are launched
        dataset.set_loader_cfg(num_workers=num_workers)

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    if len(list(dataset[0])) == 2:
        has_index = False
    elif len(list(dataset[0])) == 3:
        has_index = True
    else:
        raise ValueError

    if collate_fn is None:
        if use_prefetcher:
            if has_index:
                collate_fn = fast_collate_for_index_input_target
            else:
                collate_fn = fast_collate_for_input_target
        else:
            collate_fn = torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.

        if has_index:
            # augment index들에만 augmentation이 적용됨.
            loader = IndexInputTargetPrefetchLoader(
                loader,
                mean=mean,
                std=std,
                channels=input_size[0],
                device=device,
                fp16=fp16,  # deprecated, use img_dtype
                img_dtype=img_dtype,
                re_prob=prefetch_re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits
            )
        else:
            loader = InputTargetPrefetchLoader(
                loader,
                mean=mean,
                std=std,
                channels=input_size[0],
                device=device,
                fp16=fp16,  # deprecated, use img_dtype
                img_dtype=img_dtype,
                re_prob=prefetch_re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits
            )

    return loader

# def create_loader(
#         dataset: Union[ImageDataset, IterableImageDataset],
#         input_size: Union[int, Tuple[int, int], Tuple[int, int, int]],
#         batch_size: int,
#         is_training: bool = False,
#         no_aug: bool = False,
#         default_augment_config: dict = None,
#         special_augment_config: dict = None,
#         num_aug_repeats: int = 0,
#         num_aug_splits: int = 0,
#         mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
#         std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
#         num_workers: int = 1,
#         distributed: bool = False,
#         collate_fn: Optional[Callable] = None,
#         pin_memory: bool = False,
#         fp16: bool = False,  # deprecated, use img_dtype
#         img_dtype: torch.dtype = torch.float32,
#         device: torch.device = torch.device('cuda'),
#         use_prefetcher: bool = True,
#         use_multi_epochs_loader: bool = False,
#         persistent_workers: bool = True,
#         worker_seeding: str = 'all',
#         tf_preprocessing: bool = False,
#         augment_indices: list = None,
# ):
#     """

#     Args:
#         dataset: The image dataset to load.
#         input_size: Target input size (channels, height, width) tuple or size scalar.
#         batch_size: Number of samples in a batch.
#         is_training: Return training (random) transforms.
#         no_aug: Disable augmentation for training (useful for debug).
#         re_prob: Random erasing probability.
#         re_mode: Random erasing fill mode.
#         re_count: Number of random erasing regions.
#         re_split: Control split of random erasing across batch size.
#         scale: Random resize scale range (crop area, < 1.0 => zoom in).
#         ratio: Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).
#         hflip: Horizontal flip probability.
#         vflip: Vertical flip probability.
#         color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
#             Scalar is applied as (scalar,) * 3 (no hue).
#         color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug
#         grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
#         gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
#         auto_augment: Auto augment configuration string (see auto_augment.py).
#         num_aug_repeats: Enable special sampler to repeat same augmentation across distributed GPUs.
#         num_aug_splits: Enable mode where augmentations can be split across the batch.
#         interpolation: Image interpolation mode.
#         mean: Image normalization mean.
#         std: Image normalization standard deviation.
#         num_workers: Num worker processes per DataLoader.
#         distributed: Enable dataloading for distributed training.
#         crop_pct: Inference crop percentage (output size / resize size).
#         crop_mode: Inference crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.
#         crop_border_pixels: Inference crop border of specified # pixels around edge of original image.
#         collate_fn: Override default collate_fn.
#         pin_memory: Pin memory for device transfer.
#         fp16: Deprecated argument for half-precision input dtype. Use img_dtype.
#         img_dtype: Data type for input image.
#         device: Device to transfer inputs and targets to.
#         use_prefetcher: Use efficient pre-fetcher to load samples onto device.
#         use_multi_epochs_loader:
#         persistent_workers: Enable persistent worker processes.
#         worker_seeding: Control worker random seeding at init.
#         tf_preprocessing: Use TF 1.0 inference preprocessing for testing model ports.

#     Returns:
#         DataLoader
#     """

#     default_augment_config = {
#         're_prob': 0.,
#         're_count': 1,
#         're_mode': 'const',
#         're_split': False,
#         'train_crop_mode': None,
#         'scale': None,
#         'ratio': None,
#         'hflip': 0.5,
#         'vflip': 0.,
#         'color_jitter': 0.4,
#         'color_jitter_prob': None,
#         'grayscale_prob': 0.,
#         'gaussian_blur_prob': 0.,
#         'auto_augment': None,
#         'interpolation': 'bilinear',
#         'crop_pct': None,
#         'crop_mode': None,
#         'crop_border_pixels': None,
#     }

#     if default_augment_config is not None:
#         for attr, value in default_augment_config.items():
#             default_augment_config[attr] = value

    
#     special_augment_config = {
#         're_prob': 0.,
#         're_count': 1,
#         're_mode': 'const',
#         're_split': False,
#         'train_crop_mode': None,
#         'scale': None,
#         'ratio': None,
#         'hflip': 0.5,
#         'vflip': 0.,
#         'color_jitter': 0.4,
#         'color_jitter_prob': None,
#         'grayscale_prob': 0.,
#         'gaussian_blur_prob': 0.,
#         'auto_augment': None,
#         'interpolation': 'bilinear',
#         'crop_pct': None,
#         'crop_mode': None,
#         'crop_border_pixels': None,
#     }

#     if special_augment_config is not None:
#         for attr, value in special_augment_config.items():
#             special_augment_config[attr] = value
    
#     if augment_indices is None:

#         default_augment_config['re_num_splits'] = 0
#         if default_augment_config['re_split']:
#             # apply RE to second half of batch if no aug split otherwise line up with aug split
#             default_augment_config['re_num_splits'] = default_augment_config['num_aug_splits'] or 2
        
#         dataset.transform = create_transform(
#             input_size,
#             is_training=is_training,
#             no_aug=no_aug,
#             train_crop_mode=default_augment_config['train_crop_mode'],
#             scale=default_augment_config['scale'],
#             ratio=default_augment_config['ratio'],
#             hflip=default_augment_config['hflip'],
#             vflip=default_augment_config['vflip'],
#             color_jitter=default_augment_config['color_jitter'],
#             color_jitter_prob=default_augment_config['color_jitter_prob'],
#             grayscale_prob=default_augment_config['grayscale_prob'],
#             gaussian_blur_prob=default_augment_config['gaussian_blur_prob'],
#             auto_augment=default_augment_config['auto_augment'],
#             interpolation=default_augment_config['interpolation'],
#             mean=mean,
#             std=std,
#             crop_pct=default_augment_config['crop_pct'],
#             crop_mode=default_augment_config['crop_mode'],
#             crop_border_pixels=default_augment_config['crop_border_pixels'],
#             re_prob=default_augment_config['re_prob'],
#             re_mode=default_augment_config['re_mode'],
#             re_count=default_augment_config['re_count'],
#             re_num_splits=default_augment_config['re_num_splits'],
#             tf_preprocessing=tf_preprocessing,
#             use_prefetcher=use_prefetcher,
#             separate=num_aug_splits > 0,
#         )
#     else:
#         dataset.special_indices = set(augment_indices)

#         default_augment_config['re_num_splits'] = 0
#         if default_augment_config['re_split']:
#             # apply RE to second half of batch if no aug split otherwise line up with aug split
#             default_augment_config['re_num_splits'] = default_augment_config['num_aug_splits'] or 2

#         special_augment_config['re_num_splits'] = 0
#         if special_augment_config['re_split']:
#             # apply RE to second half of batch if no aug split otherwise line up with aug split
#             special_augment_config['re_num_splits'] = special_augment_config['num_aug_splits'] or 2

#         dataset.default_transform = create_transform(
#             input_size,
#             is_training=is_training,
#             no_aug=True,
#             train_crop_mode=default_augment_config['train_crop_mode'],
#             scale=default_augment_config['scale'],
#             ratio=default_augment_config['ratio'],
#             hflip=default_augment_config['hflip'],
#             vflip=default_augment_config['vflip'],
#             color_jitter=default_augment_config['color_jitter'],
#             color_jitter_prob=default_augment_config['color_jitter_prob'],
#             grayscale_prob=default_augment_config['grayscale_prob'],
#             gaussian_blur_prob=default_augment_config['gaussian_blur_prob'],
#             auto_augment=default_augment_config['auto_augment'],
#             interpolation=default_augment_config['interpolation'],
#             mean=mean,
#             std=std,
#             crop_pct=default_augment_config['crop_pct'],
#             crop_mode=default_augment_config['crop_mode'],
#             crop_border_pixels=default_augment_config['crop_border_pixels'],
#             re_prob=default_augment_config['re_prob'],
#             re_mode=default_augment_config['re_mode'],
#             re_count=default_augment_config['re_count'],
#             re_num_splits=default_augment_config['re_num_splits'],
#             tf_preprocessing=tf_preprocessing,
#             use_prefetcher=use_prefetcher,
#             separate=num_aug_splits > 0,
#         )

#         special_augment_config['re_num_splits'] = 0
#         if special_augment_config['re_split']:
#             # apply RE to second half of batch if no aug split otherwise line up with aug split
#             special_augment_config['re_num_splits'] = special_augment_config['num_aug_splits'] or 2

#         dataset.special_transform = create_transform(
#             input_size,
#             is_training=is_training,
#             no_aug=False,
#             train_crop_mode=special_augment_config['train_crop_mode'],
#             scale=special_augment_config['scale'],
#             ratio=special_augment_config['ratio'],
#             hflip=special_augment_config['hflip'],
#             vflip=special_augment_config['vflip'],
#             color_jitter=special_augment_config['color_jitter'],
#             color_jitter_prob=special_augment_config['color_jitter_prob'],
#             grayscale_prob=special_augment_config['grayscale_prob'],
#             gaussian_blur_prob=special_augment_config['gaussian_blur_prob'],
#             auto_augment=special_augment_config['auto_augment'],
#             interpolation=special_augment_config['interpolation'],
#             mean=mean,
#             std=std,
#             crop_pct=special_augment_config['crop_pct'],
#             crop_mode=special_augment_config['crop_mode'],
#             crop_border_pixels=special_augment_config['crop_border_pixels'],
#             re_prob=special_augment_config['re_prob'],
#             re_mode=special_augment_config['re_mode'],
#             re_count=special_augment_config['re_count'],
#             re_num_splits=special_augment_config['re_num_splits'],
#             tf_preprocessing=tf_preprocessing,
#             use_prefetcher=use_prefetcher,
#             separate=num_aug_splits > 0,
#         )

#     if isinstance(dataset, IterableImageDataset):
#         # give Iterable datasets early knowledge of num_workers so that sample estimates
#         # are correct before worker processes are launched
#         dataset.set_loader_cfg(num_workers=num_workers)

#     sampler = None
#     if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
#         if is_training:
#             if num_aug_repeats:
#                 sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
#             else:
#                 sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#         else:
#             # This will add extra duplicate entries to result in equal num
#             # of samples per-process, will slightly alter validation results
#             sampler = OrderedDistributedSampler(dataset)
#     else:
#         assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

#     # if collate_fn is None:
#     #     collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

#     if len(list(dataset[0])) == 2:
#         has_index = False
#     elif len(list(dataset[0])) == 3:
#         has_index = True
#     else:
#         raise ValueError

#     if collate_fn is None:
#         if use_prefetcher:
#             if has_index:
#                 collate_fn = fast_collate_for_index_input_target
#             else:
#                 collate_fn = fast_collate_for_input_target
#         else:
#             collate_fn = torch.utils.data.dataloader.default_collate

#     loader_class = torch.utils.data.DataLoader
#     if use_multi_epochs_loader:
#         loader_class = MultiEpochsDataLoader

#     loader_args = dict(
#         batch_size=batch_size,
#         shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
#         num_workers=num_workers,
#         sampler=sampler,
#         collate_fn=collate_fn,
#         pin_memory=pin_memory,
#         drop_last=is_training,
#         worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
#         persistent_workers=persistent_workers
#     )
#     try:
#         loader = loader_class(dataset, **loader_args)
#     except TypeError as e:
#         loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
#         loader = loader_class(dataset, **loader_args)
#     if use_prefetcher:
#         prefetch_re_prob = default_augment_config['re_prob'] if is_training and not no_aug else 0.
#         if has_index:

#             default_erasing_config = {
#                 're_prob': prefetch_re_prob,
#                 're_mode': default_augment_config['re_mode'],
#                 're_count': default_augment_config['re_count'],
#                 're_num_splits': default_augment_config['re_num_splits'] 
#             }

#             if augment_indices is not None:
#                 special_erasing_config = {
#                     're_prob': prefetch_re_prob,
#                     're_mode': special_augment_config['re_mode'],
#                     're_count': special_augment_config['re_count'],
#                     're_num_splits': special_augment_config['re_num_splits'] 
#                 }
#             else:
#                 special_erasing_config = None

#             loader = IndexInputTargetPrefetchLoader(
#                 loader,
#                 mean=mean,
#                 std=std,
#                 channels=input_size[0],
#                 device=device,
#                 fp16=fp16,  # deprecated, use img_dtype
#                 img_dtype=img_dtype,
#                 default_erase_config=default_erasing_config,
#                 special_erase_config=special_erasing_config,
#             )
#         else:
#             loader = InputTargetPrefetchLoader(
#                 loader,
#                 mean=mean,
#                 std=std,
#                 channels=input_size[0],
#                 device=device,
#                 fp16=fp16,  # deprecated, use img_dtype
#                 img_dtype=img_dtype,
#                 re_prob=prefetch_re_prob,
#                 re_mode=default_augment_config['re_mode'],
#                 re_count=default_augment_config['re_count'],
#                 re_num_splits=default_augment_config['re_num_splits']
#             )

#     return loader
    
