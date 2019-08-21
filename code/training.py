import random
import time
import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os


class Trainer():
    def __init__(self, model, training_dataset, evaluation_datasets, batch_size, opt, learning_rate, momentum,
                 epoch_count, acc_topk, print_interval, word_weights, use_weight_loss, to_cuda, log_dir):
        self.model = model
        self.training_dataset = training_dataset
        self.evaluation_datasets = evaluation_datasets
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
        self.use_weight_loss = use_weight_loss
        self.word_weights = word_weights

        self.log_file = open(os.path.join(log_dir, "log.txt"), "w")
        if self.to_cuda:
            if self.word_weights is not None:
                self.word_weights = self.word_weights.cuda()

    
    def log(self, *args, **kwargs):
        time_prefix = f"[{time.strftime('%H:%M:%S', time.localtime())}]"
        print(time_prefix, *args, **kwargs)
        print(time_prefix, *args, **kwargs, file=self.log_file)
        self.log_file.flush()


    def train(self, train_loader, loss_function, optimizer, epoch_train_loss, epoch_train_acc,
              predicted_train_sentences, ground_truth_train_sentences):
        for context_ids_batch, context_pos_batch, context_mask_batch, target_xs_batch, target_xs_mask_batch, target_ys_batch in train_loader:
            context_ids = self.batch2var(context_ids_batch, False)
            context_pos = self.batch2var(context_pos_batch, False)
            context_mask = self.batch2var(context_mask_batch, False)
            target_xs = self.batch2var(target_xs_batch, False)
            target_xs_mask = self.batch2var(target_xs_mask_batch, False)
            target_ys = self.batch2var(target_ys_batch, False)

            # feedforward - backprop
            optimizer.zero_grad()
            outputs = self.model(context_ids, context_pos, context_mask, target_xs, target_xs_mask)
            outputs_fixed, target_ys_fixed = self.fix_dimensions(outputs, target_ys)
            loss = loss_function(outputs_fixed, target_ys_fixed)
            loss.backward()
            optimizer.step()

            # train loss
            epoch_train_loss.append(loss.item())
            epoch_train_acc.append(self.compute_accuracy_topk(outputs_fixed, target_ys_fixed))
            self.print_results(context_pos[0], context_ids[0], target_xs[0], target_ys[0], outputs[0])

            if predicted_train_sentences is not None and ground_truth_train_sentences is not None:
                self.populate_predicted_and_ground_truth(predicted_train_sentences, ground_truth_train_sentences,
                                                         context_pos_batch, context_ids_batch, target_xs, target_ys,
                                                         outputs)

    def evaluate(self, eval_loader, loss_function, epoch_eval_loss, epoch_eval_acc, predicted_eval_sentences,
                 ground_truth_eval_sentences, eval_samples_for_blue_calculation):

        for context_ids_batch, context_pos_batch, context_mask_batch, target_xs_batch, target_xs_mask_batch, target_ys_batch in eval_loader:
            context_ids = self.batch2var(context_ids_batch, False)
            context_pos = self.batch2var(context_pos_batch, False)
            context_mask = self.batch2var(context_mask_batch, False)
            target_xs = self.batch2var(target_xs_batch, False)
            target_xs_mask = self.batch2var(target_xs_mask_batch, False)
            target_ys = self.batch2var(target_ys_batch, False)

            # feedforward
            outputs = self.model(context_ids, context_pos, context_mask, target_xs, target_xs_mask)
            outputs_fixed, target_ys_fixed = self.fix_dimensions(outputs, target_ys)
            loss = loss_function(outputs_fixed, target_ys_fixed)

            # train loss
            epoch_eval_loss.append(loss.item())
            epoch_eval_acc.append(self.compute_accuracy_topk(outputs_fixed, target_ys_fixed))
            self.print_results(context_pos[0], context_ids[0], target_xs[0], target_ys[0], outputs[0], True)

            if predicted_eval_sentences is not None and ground_truth_eval_sentences is not None and eval_samples_for_blue_calculation is not None:
                self.populate_predicted_and_ground_truth(predicted_eval_sentences, ground_truth_eval_sentences,
                                                         context_pos_batch, context_ids_batch, target_xs, target_ys,
                                                         outputs, eval_samples_for_blue_calculation)

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
        topk = min(self.acc_topk, outputs.shape[1])
        _, max_indices = outputs.topk(k=topk, dim=1)
        mask = torch.ones(len(target_ys))
        mask = mask.long() * -1
        if self.to_cuda:
            mask = mask.cuda()
        mask_size = (target_ys == mask).sum().item()
        # new_targets = target_ys + ((target_ys == mask).long()*-1)
        return (max_indices == target_ys.unsqueeze(dim=1)).sum().item() / (len(target_ys) - mask_size)

    def populate_predicted_and_ground_truth(self, predicted_sentences, ground_truth_sentences, context_pos, context_ids,
                                            target_pos, target_ids, predictions,
                                            eval_samples_for_blue_calculation=None):

        for cur_context_pos, cur_context_ids, cur_target_pos, cur_target_ids, cur_predictions in zip(context_pos,
                                                                                                     context_ids,
                                                                                                     target_pos,
                                                                                                     target_ids,
                                                                                                     predictions):

            orig = []
            pred = []

            i = 0
            j = 0
            pos = 0

            while pos < len(cur_context_pos):
                pred_id = None
                id = cur_context_ids[i]
                if cur_context_pos[i] == pos:
                    if id == 0:
                        break
                    i += 1
                else:
                    if j >= len(cur_target_pos):
                        self.log("error")
                        return
                    id = cur_target_ids[j]
                    if id == 0:
                        break
                    pred_id = torch.max(cur_predictions[j], dim=0)[1]
                    j += 1
                pos += 1
                if pred_id is not None:
                    orig.append(self.model.id2w[int(id.item())])
                    pred.append(self.model.id2w[int(pred_id.item() + 1)])
                else:
                    orig.append(self.model.id2w[int(id.item())])
                    pred.append(self.model.id2w[int(id.item())])
            orig_as_reference_for_blue = [orig]
            ground_truth_sentences.append(orig_as_reference_for_blue)
            predicted_sentences.append(pred)

            if eval_samples_for_blue_calculation is not None:
                eval_samples_for_blue_calculation.append(orig)

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
                    self.log("error")
                    return
                id = target_ids[j]
                if id == 0:
                    break
                pred_id = torch.max(predictions[j], dim=0)[1]
                j += 1
            pos += 1
            if pred_id is not None:
                orig += "*" + self.model.id2w[int(id.item())] + "* "
                pred += "*" + self.model.id2w[int(pred_id.item() + 1)] + "* "
            else:
                orig += self.model.id2w[int(id.item())] + " "
                pred += self.model.id2w[int(id.item())] + " "
        if is_eval:
            self.log("Eval Sample:")
        else:
            self.log("Train Sample:")
        self.log("orig: {}".format(orig))
        self.log("pred: {}".format(pred))
        self.log()

    def run(self):
        if self.use_weight_loss:
            loss_function = nn.CrossEntropyLoss(weight=self.word_weights, ignore_index=-1)  # padded outputs are ignored
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        if self.opt == "SGD":
            nestov = False
            if self.momentum > 0:
                nestov = True
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                        nesterov=nestov)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_loader = DataLoader(dataset=self.training_dataset, batch_size=self.batch_size, shuffle=True)
        if self.evaluation_datasets:
            eval_loaders = []
            for eval_dataset in self.evaluation_datasets:
                eval_loaders.append((DataLoader(dataset=eval_dataset, batch_size=self.batch_size, shuffle=False), eval_dataset.mask_ratio))
        else:
            eval_loaders = None
        train_loss_per_epoch = []
        if self.evaluation_datasets:
            eval_losses_per_epoch = [[] for _ in range(len(self.evaluation_datasets))]
        else:
            eval_losses_per_epoch = []

        for epoch in range(1, self.epoch_count + 1):
            # train
            self.model.train_model()
            epoch_train_loss = []
            epoch_train_acc = []

            calculate_blue = False # epoch == self.epoch_count # or epoch % 100 == 0

            predicted_train_sentences = None
            ground_truth_train_sentences = None
            predicted_eval_sentences = None
            ground_truth_eval_sentences = None
            eval_samples_for_blue_calculation = None

            if calculate_blue:
        #        predicted_train_sentences = []
        #        ground_truth_train_sentences = []
                predicted_eval_sentences = []
                ground_truth_eval_sentences = []
                eval_samples_for_blue_calculation = []

            self.train(train_loader, loss_function, optimizer, epoch_train_loss, epoch_train_acc,
                       predicted_train_sentences, ground_truth_train_sentences)

            # evaluate
            if eval_loaders:
                self.model.eval_model()
                epoch_eval_losses = []
                epoch_eval_accs = []
                for eval_loader in eval_loaders:
                    epoch_eval_loss = []
                    epoch_eval_acc = []
                    self.evaluate(eval_loader[0], loss_function, epoch_eval_loss, epoch_eval_acc, predicted_eval_sentences,
                              ground_truth_eval_sentences, eval_samples_for_blue_calculation)
                    epoch_eval_losses.append(epoch_eval_loss)
                    epoch_eval_accs.append(epoch_eval_acc)

            cur_train_bleu = None
            if calculate_blue:
          #     cur_train_bleu = corpus_bleu(ground_truth_train_sentences, predicted_train_sentences)
                cur_train_bleu = -1

            # compute epoch loss
            cur_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            cur_train_acc = sum(epoch_train_acc) / len(epoch_train_acc)
            train_loss_per_epoch.append(cur_train_loss)
            if eval_loaders:
                cur_eval_bleu = None
                if calculate_blue:
                    cur_eval_bleu_without_big_ref = corpus_bleu(ground_truth_eval_sentences, predicted_eval_sentences)

                    num_of_eval_sents = min(len(eval_samples_for_blue_calculation), 10000)
                    random.shuffle(eval_samples_for_blue_calculation)
                    eval_samples_for_blue_calculation = eval_samples_for_blue_calculation[:num_of_eval_sents]

                    print()
                    print(
                        "=============== Adding {} eval sentences to every reference for blue calculation ==================".format(
                            len(eval_samples_for_blue_calculation)))
                    print()
                    for gt_sent in ground_truth_eval_sentences:
                        gt_sent.extend(eval_samples_for_blue_calculation)

                    cur_eval_bleu_with_big_ref = corpus_bleu(ground_truth_eval_sentences, predicted_eval_sentences)

                cur_eval_losses = []
                cur_eval_accs = []
                for i in range(len(eval_loaders)):
                    cur_eval_losses.append(sum(epoch_eval_losses[i]) / len(epoch_eval_losses[i]))
                    cur_eval_accs.append(sum(epoch_eval_accs[i]) / len(epoch_eval_accs[i]))
                    eval_losses_per_epoch[i].append(cur_eval_losses[i])
            else:
                cur_eval_bleu = 0
                cur_eval_loss = 0
                cur_eval_acc = 0
                cur_eval_bleu_with_big_ref = 0
                cur_eval_bleu_without_big_ref = 0

            if epoch % 1 == 0 or epoch == 1:
                if calculate_blue:
                    self.log(
                        'Epoch [%d/%d] Train Loss: %.4f, Train Accuracy: %.4f, Train Bleu score: %.4f, Eval Loss: %.4f, Eval Accuracy: %.4f, Eval Bleu score (big ref): %.4f, Eval Bleu score (only gt as ref): %.4f' %
                        (epoch, self.epoch_count, cur_train_loss, cur_train_acc, cur_train_bleu, cur_eval_loss,
                         cur_eval_acc, cur_eval_bleu_with_big_ref, cur_eval_bleu_without_big_ref))
                else:
                    self.log('Epoch [%d/%d] Train Loss: %.4f, Train Accuracy: %.4f' %
                        (epoch, self.epoch_count, cur_train_loss, cur_train_acc))
                    if eval_loaders:
                        for i, eval_loader in enumerate(eval_loaders): 
                            self.log('Epoch [%d/%d] Eval Loss (%.2f): %.4f, Eval Accuracy: %.4f' %
                                (epoch, self.epoch_count, eval_loader[1], cur_eval_losses[i], cur_eval_accs[i]))
                    # self.log()

        return train_loss_per_epoch, eval_losses_per_epoch
