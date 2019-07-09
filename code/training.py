import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class Trainer():
    def __init__(self, model, training_dataset, evaluation_dataset, batch_size, learning_rate, epoch_count, to_cuda):
        self.model = model
        self.training_dataset = training_dataset
        self.evaluation_dataset = evaluation_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.to_cuda = to_cuda


    def train(self, train_loader, loss_function, optimizer, epoch_train_loss, epoch_train_acc):
        for context_batch, context_mask_batch, target_xs_batch, target_ys_batch in train_loader:
            context = self.batch2var(context_batch, True)
            context_mask = self.batch2var(context_mask_batch, False)
            target_xs = self.batch2var(target_xs_batch, True)
            target_ys = self.batch2var(target_ys_batch, False)

            # feedforward - backprop
            optimizer.zero_grad()
            outputs = self.model(context, context_mask, target_xs)
            outputs, target_ys = self.fix_dimensions(outputs, target_ys)
            loss = loss_function(outputs, target_ys)
            loss.backward()
            optimizer.step()

            # train loss
            epoch_train_loss.append(loss.item())
            epoch_train_acc.append(self.compute_accuracy(outputs, target_ys))


    def evaluate(self, eval_loader, loss_function, epoch_eval_loss, epoch_eval_acc):
        for context_batch, context_mask_batch, target_xs_batch, target_ys_batch in eval_loader:
            context = self.batch2var(context_batch, False)
            context_mask = self.batch2var(context_mask_batch, False)
            target_xs = self.batch2var(target_xs_batch, False)
            target_ys = self.batch2var(target_ys_batch, False)

            # feedforward
            outputs = self.model(context, context_mask, target_xs)
            outputs, target_ys = self.fix_dimensions(outputs, target_ys)
            loss = loss_function(outputs, target_ys)

            # train loss
            epoch_eval_loss.append(loss.item())
            epoch_eval_acc.append(self.compute_accuracy(outputs, target_ys))


    def batch2var(self, batch_param, requires_grad):
        # p = torch.stack(batch_param, dim=1).float()
        p = batch_param
        if self.to_cuda:
            p = p.cuda()

        return Variable(p, requires_grad=requires_grad)


    def fix_dimensions(self, outputs, target_ys):
        a, b, c, = outputs.shape
        outputs = outputs.reshape(a * b, c)
        a, b = target_ys.shape
        target_ys = target_ys.reshape(a * b).long()  # TODO: get long tensor in the first place
        return outputs, target_ys


    def compute_accuracy(self, outputs, target_ys):
        _, max_indices = outputs.max(dim=1)
        return (max_indices == target_ys.long()).sum() / len(target_ys)


    def run(self):
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # padded outputs will have -1 as class
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_loader = DataLoader(dataset=self.training_dataset, batch_size=self.batch_size, shuffle=True)
        if self.evaluation_dataset:
            eval_loader = DataLoader(dataset=self.evaluation_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            eval_loader = None
        train_loss_per_epoch = []
        eval_loss_per_epoch = []

        for epoch in range(1, self.epoch_count+1):
            # train
            self.model.train_model()
            epoch_train_loss = []
            epoch_train_acc = []
            self.train(train_loader, loss_function, optimizer, epoch_train_loss, epoch_train_acc)

            # evaluate
            if eval_loader:
                self.model.eval_model()
                epoch_eval_loss = []
                epoch_eval_acc = []
                self.evaluate(eval_loader, loss_function, epoch_eval_loss, epoch_eval_acc)

            # compute epoch loss
            cur_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            cur_train_acc = sum(epoch_train_acc) / len(epoch_train_acc)
            train_loss_per_epoch.append(cur_train_loss)
            if eval_loader:
                cur_eval_loss = sum(epoch_eval_loss) / len(epoch_eval_loss)
                cur_eval_acc = sum(epoch_eval_acc) / len(epoch_eval_acc)
                cur_eval_perplexity = np.exp(cur_eval_loss)
                eval_loss_per_epoch.append(cur_dev_loss)
                eval_perplexity_per_epoch(cur_eval_perplexity)
            else:
                cur_eval_loss = 0
                cur_eval_acc = 0
                cur_eval_perplexity = 0

            if epoch % 1 == 0 or epoch == 1:
                print('Epoch [%d/%d] Train Loss: %.4f, Eval Loss: %.4f' %
                      (epoch, self.epoch_count, cur_train_loss, cur_eval_loss))
                print('Epoch [%d/%d] Train Accuracy: %.4f, Eval Accuracy: %.4f, Eval Perplexity: %.4f' %
                      (epoch, self.epoch_count, cur_train_acc, cur_eval_acc, cur_eval_perplexity))
               # print()

        return train_loss_per_epoch, eval_loss_per_epoch