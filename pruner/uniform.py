import torch
import numpy as np

from .dataset_pruner import DatasetPruner


class Uniform(DatasetPruner):
    def __init__(self,
            logger,
            args,
            args_text,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb):
        super(Uniform, self).__init__(
            logger,
            args,
            args_text,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb)

        self.ratio = self.args.pruning_ratio
        self.augment_ratio = self.args.augment_ratio # augmentation ratio
        
        self.pruning_start_epoch = self.args.pruning_start_epoch
        self.pruning_end_epoch = self.args.pruning_end_epoch

        self.scores = np.ones(shape=(self.num_train_samples,))

    def before_epoch(self, epoch):

        if epoch <= self.pruning_start_epoch:
            pruned_samples = np.arange(self.num_train_samples)
            
            if self.args.augment_method == 'none':
                augment_samples = []
            elif self.args.augment_method == 'well':
                augment_samples = []
            elif self.args.augment_method == 'all':
                augment_samples = pruned_samples

        elif epoch >= self.pruning_end_epoch:
            pruned_samples = np.arange(self.num_train_samples)

            if self.args.augment_method == 'none':
                augment_samples = []
            elif self.args.augment_method == 'well':
                augment_samples = pruned_samples
            elif self.args.augment_method == 'all':
                augment_samples = pruned_samples

        else:
            # TODO: Uniform으로 랜덤하게 선택하는 기능 추가
            all_samples = np.arange(self.num_train_samples)
            pruned_samples = np.random.choice(all_samples, int(self.ratio * len(all_samples)),replace=False)

            # select samples for this epoch
            if self.args.augment_method == 'none':
                augment_samples = []
            elif self.args.augment_method == 'well':
                augment_samples = np.random.choice(pruned_samples, int(self.augment_ratio * len(pruned_samples)),replace=False)
            elif self.args.augment_method == 'all':
                augment_samples = pruned_samples
            else:
                raise NotImplementedError

        self.num_used_samples += len(pruned_samples)
        self.num_augment_samples += len(augment_samples)
        self.num_full_samples += self.num_train_samples

        self.logger.info(f"Train:{epoch:3d} Train Data Utilization: {self.num_used_samples}/{self.num_full_samples} ({self.num_used_samples/self.num_full_samples*100:.3f}%) Augmentation Data Utilization: {self.num_augment_samples}/{self.num_full_samples} ({self.num_augment_samples/self.num_full_samples*100:.3f}%)")

        return {
            'train': pruned_samples,
            'augment': augment_samples
        }
    

    def while_update(self, loss, indexes):
        # sample score update
        self.scores[indexes] = loss.detach().cpu().numpy()
        
        loss = torch.mean(loss)

        return loss