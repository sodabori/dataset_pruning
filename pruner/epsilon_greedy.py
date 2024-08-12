import time

import torch
import numpy as np

from timm import utils
from .dataset_pruner import DatasetPruner


class EpsilonGreedy(DatasetPruner):
    def __init__(self,
            logger,
            args,
            args_text,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb):
        super(EpsilonGreedy, self).__init__(
            logger,
            args,
            args_text,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb)

        self.num_epochs_per_selection = self.args.num_epoch_per_selection

        self.ratio = self.args.pruning_ratio
        self.epsilon = self.args.epsilon

        self.alpha = self.args.alpha
        
        self.pruning_start_epoch = self.args.pruning_start_epoch
        self.pruning_end_epoch = self.args.pruning_end_epoch

        self.scores = np.ones(shape=(self.num_train_samples,))

    def update_batch_score(self, loss, indices):
        # update scores
        self.scores[indices] = (1 - self.alpha) * self.scores[indices] + self.alpha * loss.detach().cpu().numpy()

    def update_scores(self, epoch):

        # disable augmentation indices
        self.train_dataset.augment_indices = []
        
        batch_time_m = utils.AverageMeter()

        self.model.eval()

        end = time.time()
        last_idx = len(self.train_loader) - 1
        with torch.no_grad():
            for batch_idx, (index, input, target) in enumerate(self.train_loader):
                last_batch = batch_idx == last_idx
                if not self.args.prefetcher:
                    input = input.to(self.device)
                    target = target.to(self.device)
                if self.args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with self.amp_autocast():
                    output = self.model(input)
                    if isinstance(output, (tuple, list)):
                        output = output[0]

                    # augmentation reduction
                    reduce_factor = self.args.tta
                    if reduce_factor > 1:
                        output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                        target = target[0:target.size(0):reduce_factor]

                    loss = self.val_loss_fn(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

                if self.args.distributed:
                    reduced_loss = utils.reduce_tensor(loss.data, self.args.world_size)
                    acc1 = utils.reduce_tensor(acc1, self.args.world_size)
                    acc5 = utils.reduce_tensor(acc5, self.args.world_size)
                else:
                    reduced_loss = loss.data

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                if self.args.distributed:
                    loss = reduced_loss

                self.update_batch_score(loss, index)

                batch_time_m.update(time.time() - end)
                end = time.time()
                if utils.is_primary(self.args) and (last_batch or batch_idx % self.args.log_interval == 0):
                    log_name = 'Score Evaluation'
                    self.logger.info(
                        f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                        f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    )

    def sample_with_epsilon_greedy(self, epoch):
        
        num_keep_samples = int(self.num_train_samples * self.ratio)
        num_random_samples = int(self.epsilon * num_keep_samples)
        num_greedy_samples = num_keep_samples - num_random_samples

        all_samples = torch.arange(self.num_train_samples)

        pruned_samples = []
        
        _, greedy_samples = torch.topk(torch.tensor(self.scores), num_greedy_samples)
        pruned_samples.extend(greedy_samples.numpy())

        non_greedy_samples = all_samples[~torch.isin(all_samples, greedy_samples)]
        selected = np.random.choice(non_greedy_samples.numpy(), num_random_samples, replace=False)

        pruned_samples.extend(selected)

        return pruned_samples, selected

    def before_epoch(self, epoch):

        if epoch <= self.pruning_start_epoch:
            self.pruned_samples = np.arange(self.num_train_samples)
            
            if self.args.augment_method == 'none':
                self.augment_samples = []
            elif self.args.augment_method == 'well':
                self.augment_samples = []
            elif self.args.augment_method == 'all':
                self.augment_samples = self.pruned_samples

        elif epoch >= self.pruning_end_epoch:
            self.pruned_samples = np.arange(self.num_train_samples)

            if self.args.augment_method == 'none':
                self.augment_samples = []
            elif self.args.augment_method == 'well':
                self.augment_samples = self.pruned_samples
            elif self.args.augment_method == 'all':
                self.augment_samples = self.pruned_samples

        else: # in pruning range
            if epoch % self.num_epochs_per_selection == 0:
                # dataset pruning with epsilon greedy
                self.pruned_samples, self.augment_samples = self.sample_with_epsilon_greedy(epoch)

                # select samples for this epoch
                if self.args.augment_method == 'none':
                    self.augment_samples = []
                elif self.args.augment_method == 'well':
                    pass # already taken
                elif self.args.augment_method == 'all':
                    self.augment_samples = self.pruned_samples
                else:
                    raise NotImplementedError

        self.num_used_samples += len(self.pruned_samples)
        self.num_augment_samples += len(self.augment_samples)
        self.num_full_samples += self.num_train_samples

        self.logger.info(f"Train:{epoch:3d} Train Data Utilization: {self.num_used_samples}/{self.num_full_samples} ({self.num_used_samples/self.num_full_samples*100:.3f}%) Augmentation Data Utilization: {self.num_augment_samples}/{self.num_full_samples} ({self.num_augment_samples/self.num_full_samples*100:.3f}%)")

        return {
            'train': self.pruned_samples,
            'augment': self.augment_samples
        }
    

    def while_update(self, loss, indexes):
        # sample score update
        self.scores[indexes] = loss.detach().cpu().numpy()
        
        loss = torch.mean(loss)

        return loss