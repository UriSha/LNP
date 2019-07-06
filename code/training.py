import torch
import torch.nn as nn
import torchvision
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


    def train(self, train_loader, loss_function, optimizer, epoch_train_loss):
        for context_batch, context_mask_batch target_xs_batch, target_ys_batch in train_loader:
            context = self.batch2var(context_batch, True)
            context_mask = self.batch2var(context_mask_batch, True)
            target_xs = self.batch2var(target_xs_batch, True)
            target_ys = self.batch2var(target_ys_batch, True)

            # feedforward - backprop
            optimizer.zero_grad()
            outputs = self.model(context, context_mask, target_xs)
            loss = loss_function(outputs, target_ys)
            loss.backward()
            optimizer.step()

            # train loss
            epoch_train_loss.append(loss.item())


    def evaluate(self, eval_loader, loss_function, epoch_eval_loss):
        for context_batch, context_mask_batch, target_xs_batch, target_ys_batch in eval_loader:
            context = self.batch2var(context_batch, False)
            context_mask = self.batch2var(context_mask_batch, False)
            target_xs = self.batch2var(target_xs_batch, False)
            target_ys = self.batch2var(target_ys_batch, False)

            # feedforward
            outputs = self.model(context, context_mask, target_xs)
            loss = loss_function(outputs, target_ys)

            # train loss
            epoch_eval_loss.append(loss.item())


    def batch2var(self, batch_param, requires_grad):
        p = torch.stack(batch_param, dim=1).float()
        if self.to_cuda:
            p = p.cuda()

        return Variable(p, requires_grad=requires_grad)


    def run(self):
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # padded outputs will have -1 as class
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        train_loader = DataLoader(dataset=self.training_dataset, batch_size=self.batch_size, shuffle=True)
        if self.evaluation_dataset:
            eval_loader = DataLoader(dataset=self.evaluation_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            eval_loader = None
        train_loss_per_epoch = []

        for epoch in range(self.epoch_count):
            # train
            self.model.train_model()
            epoch_train_loss = []
            self.train(train_loader, loss_function, optimizer, epoch_train_loss)

            # evaluate
            if eval_loader:
                self.model.eval_model()
                epoch_eval_loss = []
                self.evaluate(eval_loader, loss_function, epoch_eval_loss)

            # compute epoch loss
            cur_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            train_loss_per_epoch.append(cur_train_loss)
            cur_eval_loss = sum(epoch_eval_loss) / len(epoch_eval_loss)
            eval_loss_per_epoch.append(cur_dev_loss)

            if (epoch) % 10 == 0:
                print('Epoch [%d/%d] Train Loss: %.4f, Eval Loss: %.4f' %
                      (epoch+1, self.epoch_count, cur_train_loss, cur_eval_loss))
                print()

        return train_loss_per_epoch, eval_loss_per_epoch