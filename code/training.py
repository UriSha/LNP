import torch
import torch.nn as nn
import torchvision
import numpy as np
import time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class Trainer():
    def __init__(self, model, training_dataset, evaluation_dataset, batch_size, opt, learning_rate, momentum, epoch_count, acc_topk, print_interval, word_weights, to_cuda):
        self.model = model
        self.training_dataset = training_dataset
        self.evaluation_dataset = evaluation_dataset
        self.opt = opt
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.acc_topk = acc_topk
        self.to_cuda = to_cuda
        self.last_print_train = time.time()
        self.last_print_eval = time.time()
        self.print_interval = print_interval
        self.word_weights = word_weights
        if self.to_cuda:
            self.word_weights = self.word_weights.cuda()


    def train(self, train_loader, loss_function, optimizer, epoch_train_loss, epoch_train_acc):
        for context_ids_batch, context_pos_batch, context_mask_batch, target_xs_batch, target_ys_batch in train_loader:
            context_ids = self.batch2var(context_ids_batch, False)
            context_pos = self.batch2var(context_pos_batch, False)
            context_mask = self.batch2var(context_mask_batch, False)
            target_xs = self.batch2var(target_xs_batch, False)
            target_ys = self.batch2var(target_ys_batch, False)

            # feedforward - backprop
            optimizer.zero_grad()
            outputs = self.model(context_ids, context_pos, context_mask, target_xs)
            outputs_fixed, target_ys_fixed = self.fix_dimensions(outputs, target_ys)
            loss = loss_function(outputs_fixed, target_ys_fixed)
            loss.backward()
            optimizer.step()

            # train loss
            epoch_train_loss.append(loss.item())
            epoch_train_acc.append(self.compute_accuracy_topk(outputs_fixed, target_ys_fixed))
            self.print_results(context_pos[0], context_ids[0], target_xs[0], target_ys[0], outputs[0])


    def evaluate(self, eval_loader, loss_function, epoch_eval_loss, epoch_eval_acc):
        for context_ids_batch, context_pos_batch, context_mask_batch, target_xs_batch, target_ys_batch in eval_loader:
            context_ids = self.batch2var(context_ids_batch, False)
            context_pos = self.batch2var(context_pos_batch, False)
            context_mask = self.batch2var(context_mask_batch, False)
            target_xs = self.batch2var(target_xs_batch, False)
            target_ys = self.batch2var(target_ys_batch, False)

            # feedforward
            outputs = self.model(context_ids, context_pos, context_mask, target_xs)
            outputs_fixed, target_ys_fixed = self.fix_dimensions(outputs, target_ys)
            loss = loss_function(outputs_fixed, target_ys_fixed)

            # train loss
            epoch_eval_loss.append(loss.item())
            epoch_eval_acc.append(self.compute_accuracy_topk(outputs_fixed, target_ys_fixed))
            self.print_results(context_pos[0], context_ids[0], target_xs[0], target_ys[0], outputs[0], True)


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
        target_ys = target_ys.reshape(a * b)
        return outputs, target_ys - 1


    def compute_accuracy(self, outputs, target_ys):
        _, max_indices = outputs.max(dim=1)
        mask = torch.ones(len(target_ys)) * -1
        mask = mask.long()
        if self.to_cuda:
            mask = mask.cuda()
        mask_size = (target_ys == mask).sum()
        return (max_indices == target_ys).sum() / (len(target_ys) - mask_size)


    def compute_accuracy_topk(self, outputs, target_ys):
        _, max_indices = outputs.topk(k=self.acc_topk, dim=1)
        mask = torch.ones(len(target_ys))
        mask = mask.long() * -1
        if self.to_cuda:
            mask = mask.cuda()
        mask_size = (target_ys == mask).sum().item()
        new_targets = target_ys + ((target_ys == mask).long()*-1)
        return (max_indices == new_targets.unsqueeze(dim=1)).sum().item() / (len(target_ys) - mask_size)


    def print_results(self, context_pos, context_ids, target_pos, target_ids, predictions, is_eval=False):
        if is_eval:
            if time.time() - self.last_print_eval < self.print_interval:
                return
            self.last_print_eval = time.time()
        else:
            if time.time() - self.last_print_train < self.print_interval:
                return
            self.last_print_train = time.time()
        i = 0
        j = 0
        pos = 0
        orig = ""
        pred = ""
        while pos < len(context_pos):
            pred_id = None
            id = context_ids[i]
            if context_pos[i] == pos:
                if id == 0:
                    break
                i += 1
            else:
                if j >= len(target_pos):
                    print("error")
                    return
                id = target_ids[j]
                if id == 0:
                    break
                pred_id = torch.max(predictions[j], dim=0)[1]
                j += 1
            pos += 1
            if pred_id:
                orig += "*" + self.model.id2w[int(id.item())] + "* "
                pred += "*" + self.model.id2w[int(pred_id.item())] + "* "
            else:
                orig += self.model.id2w[int(id.item())] + " "
                pred += self.model.id2w[int(id.item())] + " "
        if is_eval:
            print("Eval Sample:")
        else:
            print("Train Sample:")
        print("orig: {}".format(orig))
        print("pred: {}".format(pred))
        print()


    def run(self):
        loss_function = nn.CrossEntropyLoss(weight=self.word_weights, ignore_index=-1)  # padded outputs are ignored
        if self.opt == "SGD":
            nestov = False
            if self.momentum > 0:
                nestov = True
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=nestov)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_loader = DataLoader(dataset=self.training_dataset, batch_size=self.batch_size, shuffle=True)
        if self.evaluation_dataset:
            eval_loader = DataLoader(dataset=self.evaluation_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            eval_loader = None
        train_loss_per_epoch = []
        eval_loss_per_epoch = []
        eval_perplexity_per_epoch = []

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
                cur_eval_perplexity = np.exp(-cur_eval_loss)
                eval_loss_per_epoch.append(cur_eval_loss)
                eval_perplexity_per_epoch.append(cur_eval_perplexity)
            else:
                cur_eval_loss = 0
                cur_eval_acc = 0
                cur_eval_perplexity = 0

            if epoch % 1 == 0 or epoch == 1:
                print('Epoch [%d/%d] Train Loss: %.4f, Train Accuracy: %.4f, Eval Loss: %.4f, Eval Accuracy: %.4f, Eval Perplexity: %.4f' %
                      (epoch, self.epoch_count, cur_train_loss, cur_train_acc, cur_eval_loss, cur_eval_acc, cur_eval_perplexity))
               # print()

        return train_loss_per_epoch, eval_loss_per_epoch