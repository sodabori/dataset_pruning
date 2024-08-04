import torch
import numpy as np

from .dataset_pruner import DatasetPruner


class InfoBatch(DatasetPruner):
    def __init__(self,
            logger,
            config_parser,
            parser,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb):
        super(InfoBatch, self).__init__(
            logger,
            config_parser,
            parser,
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
        if epoch > self.pruning_start_epoch and epoch < self.pruning_end_epoch:
            b = self.scores < (self.scores.mean() * self.multiplier)
            well_learned_samples = np.where(b)[0]

            pruned_samples = []
            pruned_samples.extend(np.where(np.invert(b))[0])
            selected = np.random.choice(well_learned_samples, int(self.ratio*len(well_learned_samples)),replace=False)

            self.reset_weights()
            if len(selected)>0:
                self.weights[selected]=1/self.ratio
                pruned_samples.extend(selected)

        else:
            pruned_samples = np.arange(self.num_train_samples)

        self.num_used_samples += len(pruned_samples)
        self.num_full_samples += self.num_train_samples

        print("scores: ", self.scores)
        # print("pruned_samples: ", pruned_samples)
        self.logger.info(f"Train:{epoch:2d} Data Utilization: {self.num_used_samples}/{self.num_full_samples} ({self.num_used_samples/self.num_full_samples*100:.3f}%)")

        return pruned_samples
    

    def while_update(self, loss, indexes):
        # sample score update
        self.scores[indexes] = loss.detach().cpu().numpy()

        # loss reweighting
        if self.rescaling:
            loss = loss * torch.tensor(self.weights[indexes]).to(device=self.device)
        
        loss = torch.mean(loss)

        return loss