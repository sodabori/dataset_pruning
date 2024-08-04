import wandb
import argparse
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

from .data.indexed_dataset import get_indexed_dataset
from .data.loader import create_loader


class DatasetPruner:
    method_name='default'
    def __init__(
            self,
            logger,
            config_parser,
            parser,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb):
        
        self.logger = logger

        utils.setup_default_logging()
        self.args, args_text = self._parse_args(config_parser, parser)

        if self.args.device_modules:
            for module in self.args.device_modules:
                importlib.import_module(module)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.args.prefetcher = not self.args.no_prefetcher
        self.args.grad_accum_steps = max(1, self.args.grad_accum_steps)
        self.device = utils.init_distributed_device(self.args)
        if self.args.distributed:
            self.logger.info(
                'Training in distributed mode with multiple processes, 1 device per process.'
                f'Process {self.args.rank}, total {self.args.world_size}, device {self.args.device}.')
        else:
            self.logger.info(f'Training with a single process on 1 device ({self.args.device}).')
        assert self.args.rank >= 0

        use_amp, amp_dtype = self.resolve_amp_arguments(has_apex, has_native_amp)

        utils.random_seed(self.args.seed, self.args.rank)

        if self.args.fuser:
                utils.set_jit_fuser(self.args.fuser)
        if self.args.fast_norm:
            set_fast_norm()

        in_chans = 3
        if self.args.in_chans is not None:
            in_chans = self.args.in_chans
        elif self.args.input_size is not None:
            in_chans = self.args.input_size[0]

        factory_kwargs = {}
        if self.args.pretrained_path:
            # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
            factory_kwargs['pretrained_cfg_overlay'] = dict(
                file=self.args.pretrained_path,
                num_classes=-1,  # force head adaptation
            )
        
        self.set_model(in_chans, has_apex, use_amp, factory_kwargs)

        self.set_learning_rate()

        self.optimizer = create_optimizer_v2(
            self.model,
            **optimizer_kwargs(cfg=self.args),
            **self.args.opt_kwargs,
        )

        self.set_amp_loss_scaling_and_op_casting(use_amp, amp_dtype)

        resume_epoch = self.maybe_set_resume()

        self.set_ema_model()

        self.set_distributed_training(has_apex, use_amp, has_compile)

        self.create_train_and_eval_datasets()

        mixup_active = self.set_mixup_and_cutmix()

        self.wrap_dataset_with_augmix_helper()

        self.create_data_loaders_with_augmentation_pipeline()
        
        self.set_loss_function(mixup_active)

        decreasing_metric = self.set_checkpoint_saver_and_eval_metric_tracking(args_text, has_wandb)

        self.set_learning_rate_schedule_and_starting_epoch(
            decreasing_metric=decreasing_metric,
            resume_epoch=resume_epoch)

        self.results = []

        # dataset pruning statistics
        self.num_used_samples = 0
        self.num_full_samples = 0
        self.num_train_samples = len(self.train_dataset)
    

    def _parse_args(self, config_parser, parser):
        # Do we have a config file to parse?
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)

        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text

    def resolve_amp_arguments(self, has_apex, has_native_amp):
        use_amp = None
        amp_dtype = torch.float16
        if self.args.amp:
            if self.args.amp_impl == 'apex':
                assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
                use_amp = 'apex'
                assert self.args.amp_dtype == 'float16'
            else:
                assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
                use_amp = 'native'
                assert self.args.amp_dtype in ('float16', 'bfloat16')
            if self.args.amp_dtype == 'bfloat16':
                amp_dtype = torch.bfloat16

        return use_amp, amp_dtype

    def set_model(self, in_chans, has_apex, use_amp, factory_kwargs):
        self.model = create_model(
            self.args.model,
            pretrained=self.args.pretrained,
            in_chans=in_chans,
            num_classes=self.args.num_classes,
            drop_rate=self.args.drop,
            drop_path_rate=self.args.drop_path,
            drop_block_rate=self.args.drop_block,
            global_pool=self.args.gp,
            bn_momentum=self.args.bn_momentum,
            bn_eps=self.args.bn_eps,
            scriptable=self.args.torchscript,
            checkpoint_path=self.args.initial_checkpoint,
            **factory_kwargs,
            **self.args.model_kwargs,
        )
        if self.args.head_init_scale is not None:
            with torch.no_grad():
                self.model.get_classifier().weight.mul_(self.args.head_init_scale)
                self.model.get_classifier().bias.mul_(self.args.head_init_scale)
        if self.args.head_init_bias is not None:
            nn.init.constant_(self.model.get_classifier().bias, self.args.head_init_bias)

        if self.args.num_classes is None:
            assert hasattr(self.model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            self.args.num_classes = self.model.num_classes  # FIXME handle model default vs config num_classes more elegantly

        if self.args.grad_checkpointing:
            self.model.set_grad_checkpointing(enable=True)

        if utils.is_primary(self.args):
            self.logger.info(
                f'Model {safe_model_name(self.args.model)} created, param count:{sum([m.numel() for m in self.model.parameters()])}')

        self.data_config = resolve_data_config(vars(self.args), model=self.model, verbose=utils.is_primary(self.args))

        # setup augmentation batch splits for contrastive loss or split bn
        self.num_aug_splits = 0
        if self.args.aug_splits > 0:
            assert self.args.aug_splits > 1, 'A split of 1 makes no sense'
            num_aug_splits = self.args.aug_splits

        # enable split bn (separate bn stats per batch-portion)
        if self.args.split_bn:
            assert self.num_aug_splits > 1 or self.args.resplit
            self.model = convert_splitbn_model(self.model, max(self.num_aug_splits, 2))

        # move model to GPU, enable channels last layout if set
        self.model.to(device=self.device)
        if self.args.channels_last:
            self.model.to(memory_format=torch.channels_last)

        # setup synchronized BatchNorm for distributed training
        if self.args.distributed and self.args.sync_bn:
            self.args.dist_bn = ''  # disable dist_bn when sync BN active
            assert not self.args.split_bn
            if has_apex and use_amp == 'apex':
                # Apex SyncBN used with Apex AMP
                # WARNING this won't currently work with models using BatchNormAct2d
                self.model = convert_syncbn_model(self.model)
            else:
                self.model = convert_sync_batchnorm(self.model)
            if utils.is_primary(self.args):
                self.logger.info(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        if self.args.torchscript:
            assert not self.args.torchcompile
            assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
            assert not self.args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            self.model = torch.jit.script(self.model)

    def set_learning_rate(self):
        if not self.args.lr:
            global_batch_size = self.args.batch_size * self.args.world_size * self.args.grad_accum_steps
            batch_ratio = global_batch_size / self.args.lr_base_size
            if not self.args.lr_base_scale:
                on = self.args.opt.lower()
                self.args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
            if self.args.lr_base_scale == 'sqrt':
                batch_ratio = batch_ratio ** 0.5
            self.args.lr = self.args.lr_base * batch_ratio
            if utils.is_primary(self.args):
                self.logger.info(
                    f'Learning rate ({self.args.lr}) calculated from base learning rate ({self.args.lr_base}) '
                    f'and effective global batch size ({global_batch_size}) with {self.args.lr_base_scale} scaling.')

    def set_amp_loss_scaling_and_op_casting(self, use_amp, amp_dtype):
        
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        if use_amp == 'apex':
            assert self.device.type == 'cuda'
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            self.loss_scaler = ApexScaler()
            if utils.is_primary(self.args):
                self.logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif use_amp == 'native':
            try:
                self.amp_autocast = partial(torch.autocast, device_type=self.device.type, dtype=amp_dtype)
            except (AttributeError, TypeError):
                # fallback to CUDA only AMP for PyTorch < 1.10
                assert self.device.type == 'cuda'
                self.amp_autocast = torch.cuda.amp.autocast
            if self.device.type == 'cuda' and amp_dtype == torch.float16:
                # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
                self.loss_scaler = NativeScaler()
            if utils.is_primary(self.args):
                self.logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            if utils.is_primary(self.args):
                self.logger.info('AMP not enabled. Training in float32.')

    def maybe_set_resume(self):
        resume_epoch = None
        if self.args.resume:
            resume_epoch = resume_checkpoint(
                self.model,
                self.args.resume,
                optimizer=None if self.args.no_resume_opt else self.optimizer,
                loss_scaler=None if self.args.no_resume_opt else self.loss_scaler,
                log_info=utils.is_primary(self.args),
            )

        return resume_epoch

    def set_ema_model(self):
        self.model_ema = None
        if self.args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
            model_ema = utils.ModelEmaV3(
                self.model,
                decay=self.args.model_ema_decay,
                use_warmup=self.args.model_ema_warmup,
                device='cpu' if self.args.model_ema_force_cpu else None,
            )
            if self.args.resume:
                load_checkpoint(model_ema.module, self.args.resume, use_ema=True)
            if self.args.torchcompile:
                model_ema = torch.compile(model_ema, backend=self.args.torchcompile)

    def set_distributed_training(self, has_apex, use_amp, has_compile):
        if self.args.distributed:
            if has_apex and use_amp == 'apex':
                # Apex DDP preferred unless native amp is activated
                if utils.is_primary(self.args):
                    self.logger.info("Using NVIDIA APEX DistributedDataParallel.")
                model = ApexDDP(self.model, delay_allreduce=True)
            else:
                if utils.is_primary(self.args):
                    self.logger.info("Using native Torch DistributedDataParallel.")
                model = NativeDDP(self.model, device_ids=[self.device], broadcast_buffers=not self.args.no_ddp_bb)
            # NOTE: EMA model does not need to be wrapped by DDP

        if self.args.torchcompile:
            # torch compile should be done after DDP
            assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
            model = torch.compile(self.model, backend=self.args.torchcompile)

    def create_train_and_eval_datasets(self):
        if self.args.data and not self.args.data_dir:
            self.args.data_dir = self.args.data
        if self.args.input_img_mode is None:
            input_img_mode = 'RGB' if self.data_config['input_size'][0] == 3 else 'L'
        else:
            input_img_mode = self.args.input_img_mode

        args = [self.args.dataset]

        kwargs = {
            'root': self.args.data_dir,
            'split': self.args.train_split,
            'is_training': True,
            'class_map': self.args.class_map,
            'download': self.args.dataset_download,
            'batch_size': self.args.batch_size,
            'seed': self.args.seed,
            'repeats': self.args.epoch_repeats,
            'input_img_mode': input_img_mode,
            'input_key': self.args.input_key,
            'target_key': self.args.target_key,
            'num_samples': self.args.train_num_samples,
        }

        # train_dataset = create_dataset(
        #     self.args.dataset,
        #     root=self.args.data_dir,
        #     split=self.args.train_split,
        #     is_training=True,
        #     class_map=self.args.class_map,
        #     download=self.args.dataset_download,
        #     batch_size=self.args.batch_size,
        #     seed=self.args.seed,
        #     repeats=self.args.epoch_repeats,
        #     input_img_mode=input_img_mode,
        #     input_key=self.args.input_key,
        #     target_key=self.args.target_key,
        #     num_samples=self.args.train_num_samples,
        # )

        train_dataset = create_dataset(*args, **kwargs)

        self.train_dataset = get_indexed_dataset(train_dataset, *args, **kwargs)

        if self.args.val_split:
            self.val_dataset = create_dataset(
                self.args.dataset,
                root=self.args.data_dir,
                split=self.args.val_split,
                is_training=False,
                class_map=self.args.class_map,
                download=self.args.dataset_download,
                batch_size=self.args.batch_size,
                input_img_mode=input_img_mode,
                input_key=self.args.input_key,
                target_key=self.args.target_key,
                num_samples=self.args.val_num_samples,
            )

    def set_mixup_and_cutmix(self):
        self.collate_fn = None
        self.mixup_fn = None
        mixup_active = self.args.mixup > 0 or self.args.cutmix > 0. or self.args.cutmix_minmax is not None
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=self.args.mixup,
                cutmix_alpha=self.args.cutmix,
                cutmix_minmax=self.args.cutmix_minmax,
                prob=self.args.mixup_prob,
                switch_prob=self.args.mixup_switch_prob,
                mode=self.args.mixup_mode,
                label_smoothing=self.args.smoothing,
                num_classes=self.args.num_classes
            )
            if self.args.prefetcher:
                assert not self.num_aug_splits  # collate conflict (need to support de-interleaving in collate mixup)
                self.collate_fn = FastCollateMixup(**mixup_args)
            else:
                self.mixup_fn = Mixup(**mixup_args)

        return mixup_active

    def wrap_dataset_with_augmix_helper(self):
        if self.num_aug_splits > 1:
            dataset_train = AugMixDataset(dataset_train, num_splits=self.num_aug_splits)

    def create_data_loaders_with_augmentation_pipeline(self):
        train_interpolation = self.args.train_interpolation
        if self.args.no_aug or not train_interpolation:
            train_interpolation = self.data_config['interpolation']
        self.train_loader = create_loader(
            self.train_dataset,
            input_size=self.data_config['input_size'],
            batch_size=self.args.batch_size,
            is_training=True,
            no_aug=self.args.no_aug,
            re_prob=self.args.reprob,
            re_mode=self.args.remode,
            re_count=self.args.recount,
            re_split=self.args.resplit,
            train_crop_mode=self.args.train_crop_mode,
            scale=self.args.scale,
            ratio=self.args.ratio,
            hflip=self.args.hflip,
            vflip=self.args.vflip,
            color_jitter=self.args.color_jitter,
            color_jitter_prob=self.args.color_jitter_prob,
            grayscale_prob=self.args.grayscale_prob,
            gaussian_blur_prob=self.args.gaussian_blur_prob,
            auto_augment=self.args.aa,
            num_aug_repeats=self.args.aug_repeats,
            num_aug_splits=self.num_aug_splits,
            interpolation=train_interpolation,
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            num_workers=self.args.workers,
            distributed=self.args.distributed,
            collate_fn=self.collate_fn,
            pin_memory=self.args.pin_mem,
            device=self.device,
            use_prefetcher=self.args.prefetcher,
            use_multi_epochs_loader=self.args.use_multi_epochs_loader,
            worker_seeding=self.args.worker_seeding,
        )

        self.val_loader = None
        if self.args.val_split:
            eval_workers = self.args.workers
            if self.args.distributed and ('tfds' in self.args.dataset or 'wds' in self.args.dataset):
                # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
                eval_workers = min(2, self.args.workers)
            self.val_loader = create_loader(
                self.val_dataset,
                input_size=self.data_config['input_size'],
                batch_size=self.args.validation_batch_size or self.args.batch_size,
                is_training=False,
                interpolation=self.data_config['interpolation'],
                mean=self.data_config['mean'],
                std=self.data_config['std'],
                num_workers=eval_workers,
                distributed=self.args.distributed,
                crop_pct=self.data_config['crop_pct'],
                pin_memory=self.args.pin_mem,
                device=self.device,
                use_prefetcher=self.args.prefetcher,
            )

    def set_loss_function(self, mixup_active):

        if self.args.jsd_loss:
            assert self.num_aug_splits > 1  # JSD only valid with aug splits set
            self.train_loss_fn = JsdCrossEntropy(num_splits=self.num_aug_splits, smoothing=self.args.smoothing)
        elif mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if self.args.bce_loss:
                self.train_loss_fn = BinaryCrossEntropy(
                    target_threshold=self.args.bce_target_thresh,
                    sum_classes=self.args.bce_sum,
                    pos_weight=self.args.bce_pos_weight,
                )
            else:
                self.train_loss_fn = SoftTargetCrossEntropy()
        elif self.args.smoothing:
            if self.args.bce_loss:
                self.train_loss_fn = BinaryCrossEntropy(
                    smoothing=self.args.smoothing,
                    target_threshold=self.args.bce_target_thresh,
                    sum_classes=self.args.bce_sum,
                    pos_weight=self.args.bce_pos_weight,
                )
            else:
                self.train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self.args.smoothing)
        else:
            self.train_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.train_loss_fn = self.train_loss_fn.to(device=self.device)
        self.val_loss_fn = nn.CrossEntropyLoss().to(device=self.device)

    def set_checkpoint_saver_and_eval_metric_tracking(self, args_text, has_wandb):

        self.eval_metric = self.args.eval_metric if self.val_loader is not None else 'loss'
        decreasing_metric = self.eval_metric == 'loss'
        self.best_metric = None
        self.best_epoch = None
        self.saver = None
        self.output_dir = None
        if utils.is_primary(self.args):
            if self.args.experiment:
                exp_name = self.args.experiment
            else:
                exp_name = '-'.join([
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(self.args.model),
                    str(self.data_config['input_size'][-1])
                ])
            self.output_dir = utils.get_outdir(self.args.output if self.args.output else './output/train', exp_name)
            self.saver = utils.CheckpointSaver(
                model=self.model,
                optimizer=self.optimizer,
                args=self.args,
                model_ema=self.model_ema,
                amp_scaler=self.loss_scaler,
                checkpoint_dir=self.output_dir,
                recovery_dir=self.output_dir,
                decreasing=decreasing_metric,
                max_history=self.args.checkpoint_hist
            )
            with open(os.path.join(self.output_dir, 'args.yaml'), 'w') as f:
                f.write(args_text)

        if utils.is_primary(self.args) and self.args.log_wandb:
            if has_wandb:
                wandb.init(project=self.args.experiment, config=self.args)
            else:
                self.logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`")
        
        return decreasing_metric

    def set_learning_rate_schedule_and_starting_epoch(self, decreasing_metric, resume_epoch):
        
        self.updates_per_epoch = (len(self.train_loader) + self.args.grad_accum_steps - 1) // self.args.grad_accum_steps
        self.lr_scheduler, self.num_epochs = create_scheduler_v2(
            self.optimizer,
            **scheduler_kwargs(self.args, decreasing_metric=decreasing_metric),
            updates_per_epoch=self.updates_per_epoch,
        )
        self.start_epoch = 0
        if self.args.start_epoch is not None:
            # a specified start_epoch will always override the resume epoch
            self.start_epoch = self.args.start_epoch
        elif resume_epoch is not None:
            self.start_epoch = resume_epoch
        if self.lr_scheduler is not None and self.start_epoch > 0:
            if self.args.sched_on_updates:
                self.lr_scheduler.step_update(self.start_epoch * self.updates_per_epoch)
            else:
                self.lr_scheduler.step(self.start_epoch)

        if utils.is_primary(self.args):
            self.logger.info(
                f'Scheduled epochs: {self.num_epochs}. LR stepped per {"epoch" if self.lr_scheduler.t_in_epochs else "update"}.')


    def before_run(self):
        pass

    def after_run(self):
        pass

    def before_epoch(self, epoch):
        
        self.num_used_samples += self.num_train_samples
        self.num_full_samples += self.num_train_samples

        return None

    def after_epoch(self, epoch):
        pass

    def before_batch(self):
        pass

    def after_batch(self):
        pass

    def while_update(self, loss, indexes=None):
        loss = torch.mean(loss)
        return loss

    def create_pruned_loader(self, train_indices=None):

        if train_indices is None:
            pruned_dataset = self.train_dataset
        else:
            pruned_dataset = torch.utils.data.Subset(self.train_dataset, train_indices)

        train_interpolation = self.args.train_interpolation
        if self.args.no_aug or not train_interpolation:
            train_interpolation = self.data_config['interpolation']
        self.train_loader = create_loader(
            pruned_dataset,
            input_size=self.data_config['input_size'],
            batch_size=self.args.batch_size,
            is_training=True,
            no_aug=self.args.no_aug,
            re_prob=self.args.reprob,
            re_mode=self.args.remode,
            re_count=self.args.recount,
            re_split=self.args.resplit,
            train_crop_mode=self.args.train_crop_mode,
            scale=self.args.scale,
            ratio=self.args.ratio,
            hflip=self.args.hflip,
            vflip=self.args.vflip,
            color_jitter=self.args.color_jitter,
            color_jitter_prob=self.args.color_jitter_prob,
            grayscale_prob=self.args.grayscale_prob,
            gaussian_blur_prob=self.args.gaussian_blur_prob,
            auto_augment=self.args.aa,
            num_aug_repeats=self.args.aug_repeats,
            num_aug_splits=self.num_aug_splits,
            interpolation=train_interpolation,
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            num_workers=self.args.workers,
            distributed=self.args.distributed,
            collate_fn=self.collate_fn,
            pin_memory=self.args.pin_mem,
            device=self.device,
            use_prefetcher=self.args.prefetcher,
            use_multi_epochs_loader=self.args.use_multi_epochs_loader,
            worker_seeding=self.args.worker_seeding,
        )

    def train_one_epoch(
            self,
            epoch,
            model,
            loader,
            optimizer,
            loss_fn,
            args,
            device,
            lr_scheduler,
            saver=None,
            output_dir=None,
            amp_autocast=suppress,
            loss_scaler=None,
            model_ema=None,
            mixup_fn=None,
            num_updates_total=None):

        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            if args.prefetcher and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False

        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        has_no_sync = hasattr(model, "no_sync")
        update_time_m = utils.AverageMeter()
        data_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()

        model.train()

        accum_steps = args.grad_accum_steps
        last_accum_steps = len(loader) % accum_steps
        updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
        num_updates = epoch * updates_per_epoch
        last_batch_idx = len(loader) - 1
        last_batch_idx_to_accum = len(loader) - last_accum_steps

        data_start_time = update_start_time = time.time()
        optimizer.zero_grad()
        update_sample_count = 0
        # for batch_idx, (input, target) in enumerate(loader):
        for batch_idx, (index, input, target) in enumerate(loader):

            self.before_batch()

            last_batch = batch_idx == last_batch_idx
            need_update = last_batch or (batch_idx + 1) % accum_steps == 0
            update_idx = batch_idx // accum_steps
            if batch_idx >= last_batch_idx_to_accum:
                accum_steps = last_accum_steps

            if not args.prefetcher:
                input, target = input.to(device), target.to(device)
                if mixup_fn is not None:
                    input, target = mixup_fn(input, target)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # multiply by accum steps to get equivalent for full update
            data_time_m.update(accum_steps * (time.time() - data_start_time))

            def _forward():
                with amp_autocast():
                    output = model(input)
                    loss = loss_fn(output, target)
                if accum_steps > 1:
                    loss /= accum_steps
                return loss

            def _backward(_loss):
                if loss_scaler is not None:
                    loss_scaler(
                        _loss,
                        optimizer,
                        clip_grad=args.clip_grad,
                        clip_mode=args.clip_mode,
                        parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                        create_graph=second_order,
                        need_update=need_update,
                    )
                else:
                    _loss.backward(create_graph=second_order)
                    if need_update:
                        if args.clip_grad is not None:
                            utils.dispatch_clip_grad(
                                model_parameters(model, exclude_head='agc' in args.clip_mode),
                                value=args.clip_grad,
                                mode=args.clip_mode,
                            )
                        optimizer.step()

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss = _forward()
                    loss = self.while_update(loss, index)
                    _backward(loss)
            else:
                loss = _forward()
                loss = self.while_update(loss, index)
                _backward(loss)

            if not args.distributed:
                losses_m.update(loss.item() * accum_steps, input.size(0))
            update_sample_count += input.size(0)

            if not need_update:
                data_start_time = time.time()
                continue

            num_updates += 1
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model, step=num_updates)

            if args.synchronize_step and device.type == 'cuda':
                torch.cuda.synchronize()

            self.after_batch()

            time_now = time.time()
            update_time_m.update(time.time() - update_start_time)
            update_start_time = time_now

            if update_idx % args.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if args.distributed:
                    reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                    losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                    update_sample_count *= args.world_size

                if utils.is_primary(args):
                    self.logger.info(
                        f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                        f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                        f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                        f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                        f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                        f'LR: {lr:.3e}  '
                        f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                    )

                    if args.save_images and output_dir:
                        torchvision.utils.save_image(
                            input,
                            os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                            padding=0,
                            normalize=True
                        )

            if saver is not None and args.recovery_interval and (
                    (update_idx + 1) % args.recovery_interval == 0):
                saver.save_recovery(epoch, batch_idx=update_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            update_sample_count = 0
            data_start_time = time.time()
            # end for

        if hasattr(self.optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])


    def validate(
            self,
            model,
            loader,
            loss_fn,
            args,
            device=torch.device('cuda'),
            amp_autocast=suppress,
            log_suffix=''):

        batch_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()
        top1_m = utils.AverageMeter()
        top5_m = utils.AverageMeter()

        self.model.eval()

        end = time.time()
        last_idx = len(loader) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                if not args.prefetcher:
                    input = input.to(device)
                    target = target.to(device)
                if self.args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)
                    if isinstance(output, (tuple, list)):
                        output = output[0]

                    # augmentation reduction
                    reduce_factor = args.tta
                    if reduce_factor > 1:
                        output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                        target = target[0:target.size(0):reduce_factor]

                    loss = loss_fn(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

                if args.distributed:
                    reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                    acc1 = utils.reduce_tensor(acc1, args.world_size)
                    acc5 = utils.reduce_tensor(acc5, args.world_size)
                else:
                    reduced_loss = loss.data

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                    log_name = 'Test' + log_suffix
                    self.logger.info(
                        f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                        f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                        f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                        f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                        f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                    )

        metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

        return metrics

    def run(self):

        self.before_run()

        for epoch in range(self.start_epoch, self.num_epochs):

            train_indices = self.before_epoch(epoch)

            self.create_pruned_loader(train_indices)

            if hasattr(self.train_dataset, 'set_epoch'):
                self.train_dataset.set_epoch(epoch)
            elif self.args.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_one_epoch(
                epoch,
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                loss_fn=self.train_loss_fn,
                args=self.args,
                device=self.device,
                lr_scheduler=self.lr_scheduler,
                saver=self.saver,
                output_dir=self.output_dir,
                amp_autocast=self.amp_autocast,
                loss_scaler=self.loss_scaler,
                model_ema=self.model_ema,
                mixup_fn=self.mixup_fn,
                num_updates_total=self.num_epochs * self.updates_per_epoch)

            if self.args.distributed and self.args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(self.args):
                    self.logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(self.model, self.args.world_size, self.args.dist_bn == 'reduce')

            if self.val_loader is not None:
                eval_metrics = self.validate(
                    model=self.model,
                    loader=self.val_loader,
                    loss_fn=self.val_loss_fn,
                    args=self.args,
                    device=self.device,
                    amp_autocast=self.amp_autocast,
                    log_suffix='')

                if self.model_ema is not None and not self.args.model_ema_force_cpu:
                    if self.args.distributed and self.args.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(self.model_ema, self.args.world_size, self.args.dist_bn == 'reduce')

                    ema_eval_metrics = self.validate(
                        model=self.model_ema,
                        loader=self.val_loader,
                        loss_fn=self.val_loss_fn,
                        args=self.args,
                        device=self.device,
                        amp_autocast=self.amp_autocast,
                        log_suffix=' (EMA)',
                    )
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if self.output_dir is not None:
                lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(self.output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=self.best_metric is None,
                    log_wandb=self.args.log_wandb and self.has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[self.eval_metric]
            else:
                latest_metric = train_metrics[self.eval_metric]

            if self.saver is not None:
                # save proper checkpoint with eval metric
                self.best_metric, self.best_epoch = self.saver.save_checkpoint(epoch, metric=latest_metric)

            if self.lr_scheduler is not None:
                # step LR for next epoch
                self.lr_scheduler.step(epoch + 1, latest_metric)

            self.results.append({
                'epoch': epoch,
                'train': train_metrics,
                'validation': eval_metrics,
            })

            self.after_epoch(epoch)

        self.after_run()

        self.results = {'all': self.results}
        if self.best_metric is not None:
            self.results['best'] = self.results['all'][self.best_epoch - self.start_epoch]
            self.logger.info('*** Best metric: {0} (epoch {1})'.format(self.best_metric, self.best_epoch))
        print(f'--result\n{json.dumps(self.results, indent=4)}')