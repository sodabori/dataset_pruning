import torch
import numpy as np

from .dataset_pruner import DatasetPruner


class InfoBatch(DatasetPruner):
    def __init__(self,
            logger,
            args,
            args_text,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb):
        super(InfoBatch, self).__init__(
            logger,
            args,
            args_text,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb)
        
        # infobatch related arguments
        self.ratio = self.args.pruning_ratio
        self.rescaling = self.args.rescaling
        self.multiplier = self.args.multiplier
        
        self.pruning_start_epoch = self.args.pruning_start_epoch
        self.pruning_end_epoch = self.args.pruning_end_epoch

        self.scores = np.ones(shape=(self.num_train_samples,))
        self.weights = np.ones(shape=(self.num_train_samples,))

    
    def reset_weights(self):
        self.weights = np.ones(shape=(self.num_train_samples,))


    def before_epoch(self, epoch):

        # select samples for this epoch
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
            b = self.scores < (self.scores.mean() * self.multiplier)
            well_learned_samples = np.where(b)[0]

            pruned_samples = []
            pruned_samples.extend(np.where(np.invert(b))[0])
            selected = np.random.choice(well_learned_samples, int(self.ratio*len(well_learned_samples)),replace=False)

            self.reset_weights()
            if len(selected)>0:
                self.weights[selected]=1/self.ratio
                pruned_samples.extend(selected)

            if self.args.augment_method == 'none':
                augment_samples = []
            elif self.args.augment_method == 'well':
                augment_samples = selected
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

    # def before_epoch(self, epoch): # For DEBUG

    #     # select samples for this epoch
    #     pruned_samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    #     augment_samples = []

    #     return {
    #         'train': pruned_samples,
    #         'augment': augment_samples
    #     }
    

    def while_update(self, loss, indexes):
        # sample score update
        self.scores[indexes] = loss.detach().cpu().numpy()

        # loss reweighting
        if self.rescaling:
            loss = loss * torch.tensor(self.weights[indexes]).to(device=self.device)
        
        loss = torch.mean(loss)

        return loss