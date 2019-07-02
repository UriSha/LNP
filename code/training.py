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

    def train(self):
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # padded indices will have -1 value
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        train_loader = DataLoader(
            dataset=self.training_dataset, batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(
            dataset=self.evaluation_dataset, batch_size=self.batch_size)
        train_loss_per_epoch = []

        for epoch in range(self.epoch_count):
            # train
            self.model.train_model()
            epoch_train_loss = []
            for context_batch, target_xs_batch, target_ys_batch in train_loader:
                context = torch.stack(context_batch, dim=1).float()
                target_xs = torch.stack(target_xs_batch, dim=1).float()
                target_ys = torch.stack(target_ys_batch, dim=1).float()
                if self.to_cuda:
                    context = context.cuda()
                    target_xs = target_xs.cuda()
                    target_ys = target_ys.cuda()

                context = Variable(context, requires_grad=True)
                target_xs = Variable(target_xs, requires_grad=True)
                target_ys = Variable(target_ys, requires_grad=True)

                # feedforward - backprop
                optimizer.zero_grad()
                outputs = self.model(context, target_xs)
                loss = loss_function(outputs, target_ys)
                loss.backward()
                optimizer.step()

                # train loss
                epoch_train_loss.append(loss.item())

            # evaluate
            self.model.eval_model()
            epoch_eval_loss = []
            for context_batch, target_xs_batch, target_ys_batch in eval_loader:
                context = torch.stack(context_batch, dim=1).float()
                target_xs = torch.stack(target_xs_batch, dim=1).float()
                target_ys = torch.stack(target_ys_batch, dim=1).float()
                if self.to_cuda:
                    context = context.cuda()
                    target_xs = target_xs.cuda()
                    target_ys = target_ys.cuda()

                context = Variable(context, requires_grad=False)
                target_xs = Variable(target_xs, requires_grad=False)
                target_ys = Variable(target_ys, requires_grad=False)

                # feedforward
                outputs = self.model(context, target_xs)
                loss = loss_function(outputs, target_ys)

                # train loss
                epoch_eval_loss.append(loss.item())

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