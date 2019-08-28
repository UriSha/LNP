import time
import torch
import torch.nn as nn
from sampler import Sampler
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.utils.data import DataLoader
from blue import corpus_bleu_with_joint_refrences


class Trainer():
    def __init__(self, model, train_dataset, test_datasets, tags, batch_size, opt, learning_rate, momentum,
                 epoch_count, acc_topk, print_interval, bleu_sents, to_cuda, logger, id2w):
        self.model = model
        self.epoch_count = epoch_count
        self.acc_topk = acc_topk
        self.last_print_train = time.time()
        self.last_print_tests = []
        for _ in test_datasets:
            self.last_print_tests.append(time.time())
        self.tags = tags
        self.bleu_sents = bleu_sents
        self.to_cuda = to_cuda
        self.logger = logger
        self.id2w = id2w

        self.train_sampler = Sampler("Train", print_interval, id2w, logger)
        self.test_samplers = []
        for tag in tags:
            self.test_samplers.append(Sampler(f"Test({tag:.2f})", print_interval, id2w, logger))

        self.loss_function = nn.CrossEntropyLoss(ignore_index=-1)

        if opt == "SGD":
            nestov = False
            if momentum > 0:
                nestov = True
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=nestov)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loaders = []
        for test_dataset in test_datasets:
                self.test_loaders.append(DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False))


    def run(self):

        train_loss_per_epoch = []
        test_losses_per_epoch = [[] for _ in range(len(self.test_loaders))]

        for epoch in range(1, self.epoch_count + 1):

            calculate_blue = epoch == self.epoch_count # or epoch % 100 == 0


            predicted_test_sentences = [None for _ in range(len(self.test_loaders))]
            ground_truth_test_sentences = [None for _ in range(len(self.test_loaders))]

            if calculate_blue:
                predicted_test_sentences = [[] for _ in range(len(self.test_loaders))]
                ground_truth_test_sentences = [[] for _ in range(len(self.test_loaders))]


            # Train
            self.model.train_model()
            cur_train_loss, cur_train_accuracy = self.run_epoch(self.train_loader, self.train_sampler, None, None)
            train_loss_per_epoch.append(cur_train_loss)

            # Test
            self.model.test_model()
            cur_test_losses = [None] * len(self.test_loaders)
            cur_test_accuracies = [None] * len(self.test_loaders)
            for i, test_loader in enumerate(self.test_loaders):
                cur_test_loss, cur_test_accuracy = self.run_epoch(test_loader, self.test_samplers[i], predicted_test_sentences[i], ground_truth_test_sentences[i], False)
                test_losses_per_epoch[i].append(cur_test_loss)
                cur_test_losses[i] = cur_test_loss
                cur_test_accuracies[i] = cur_test_accuracy
            

            cur_train_bleu = None
            if calculate_blue:
                # cur_train_bleu = corpus_bleu(ground_truth_train_sentences, predicted_train_sentences)
                cur_train_bleu = -1

            cur_eval_bleu = None
            if calculate_blue:

                cur_eval_bleu_without_big_ref = []
                cur_eval_bleu_with_big_ref = []
                for i, test_loader in enumerate(self.test_loaders):


                    cur_eval_bleu_without_big_ref.append(corpus_bleu(ground_truth_test_sentences[i], predicted_test_sentences[i]))


                    self.logger.log(
                        "Calculating blue score for (%.2f) with total of %d references" % (self.tags[i],len(self.bleu_sents) + len(ground_truth_test_sentences[i])))
                    cur_eval_bleu_with_big_ref.append(corpus_bleu_with_joint_refrences(self.bleu_sents, ground_truth_test_sentences[i], predicted_test_sentences[i]))

                self.logger.log("")


            if epoch % 1 == 0 or epoch == 1:
                self.logger.log(f"Epoch [{epoch}/{self.epoch_count}] Train Loss: {cur_train_loss:.4f}, Train {self.__print_accuracy(cur_train_accuracy)}")

                for i in range(len(self.test_loaders)):
                    msg = f"Epoch [{epoch}/{self.epoch_count}] Test Loss({self.tags[i]:.2f}): {cur_test_losses[i]:.4f}, Test {self.__print_accuracy(cur_test_accuracies[i])}"
                    if calculate_blue:
                        msg += f", Test Bleu score (big ref): {cur_eval_bleu_with_big_ref[i][1]:.4f}, Test Bleu Score (only gold as ref): {cur_eval_bleu_without_big_ref[i]:.4f}"
                    self.logger.log(msg)

        return train_loss_per_epoch, test_losses_per_epoch


    def run_epoch(self, loader, sampler, predicted_sentences, ground_truth_sentences, is_train=True):

        losses = []
        accuracies = []

        for context_xs_batch, context_ys_batch, context_mask_batch, target_xs_batch, target_ys_batch, target_mask_batch in loader:
            context_xs = self.__batch2var(context_xs_batch)
            context_ys = self.__batch2var(context_ys_batch)
            context_mask = self.__batch2var(context_mask_batch)
            target_xs = self.__batch2var(target_xs_batch)
            target_ys = self.__batch2var(target_ys_batch)
            target_mask = self.__batch2var(target_mask_batch)

            # feedforward - backprop
            if is_train:
                self.optimizer.zero_grad()
            
            outputs = self.model(context_xs, context_ys, context_mask, target_xs, target_mask)
            outputs_adjusted, target_ys_adjusted = self.__adjust_dimensions(outputs, target_ys)
            loss = self.loss_function(outputs_adjusted, target_ys_adjusted)

            if is_train:
                loss.backward()
                self.optimizer.step()

            
            losses.append(loss.item())
            accuracies.append(self.__compute_accuracy_topk(outputs_adjusted, target_ys_adjusted))
            sampler.sample(context_xs[0], context_ys[0], target_xs[0], target_ys[0], outputs[0])

            if predicted_sentences is not None and ground_truth_sentences is not None:
                self.__populate_predicted_and_ground_truth(predicted_sentences, ground_truth_sentences,
                                                         context_xs_batch, context_ys_batch, target_xs, target_ys,
                                                         outputs)
        
        epoch_loss = sum(losses) / len(losses)
        epoch_accuracy = []
        for i in range(len(self.acc_topk)):
            accuracy_i = [accuracy[i] for accuracy in accuracies]
            epoch_accuracy.append(sum(accuracy_i) / len(accuracy_i))
        return epoch_loss, epoch_accuracy


    def __batch2var(self, batch_param, requires_grad=False):
        if self.to_cuda:
            batch_param = batch_param.cuda()

        return Variable(batch_param, requires_grad=requires_grad)


    def __adjust_dimensions(self, outputs, target_ys):
        a, b, c, = outputs.shape
        outputs = outputs.reshape(a * b, c)
        a, b = target_ys.shape
        target_ys = target_ys.reshape(a * b)
        return outputs, target_ys - 1


    def __compute_accuracy_topk(self, outputs, target_ys):
        topks = [min(topk, outputs.shape[1]) for topk in self.acc_topk]
        results = []
        for topk in topks:  
            _, max_indices = outputs.topk(k=topk, dim=1)
            mask = torch.ones(len(target_ys))
            mask = mask.long() * -1
            if self.to_cuda:
                mask = mask.cuda()
            mask_size = (target_ys == mask).sum().item()
            # new_targets = target_ys + ((target_ys == mask).long()*-1)
            results.append((max_indices == target_ys.unsqueeze(dim=1)).sum().item() / (len(target_ys) - mask_size))
        return results


    def __print_accuracy(self, accuracies):
        accs = []
        for i in range(len(self.acc_topk)):
            accs.append(f"Top-{self.acc_topk[i]}: {accuracies[i]:.3f}")
        acc_str = ", ".join(accs)
        res = f"Accuracy: [{acc_str}]"
        return res


    def __populate_predicted_and_ground_truth(self, predicted_sentences, ground_truth_sentences, context_pos, context_ids,
                                            target_pos, target_ids, predictions):

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
                        self.logger.log("error")
                        return
                    id = cur_target_ids[j]
                    if id == 0:
                        break
                    pred_id = torch.max(cur_predictions[j], dim=0)[1]
                    j += 1
                pos += 1
                if pred_id is not None:
                    orig.append(self.id2w[int(id.item())])
                    pred.append(self.id2w[int(pred_id.item() + 1)])
                else:
                    orig.append(self.id2w[int(id.item())])
                    pred.append(self.id2w[int(id.item())])
            orig_as_reference_for_blue = [orig]
            ground_truth_sentences.append(orig_as_reference_for_blue)
            predicted_sentences.append(pred)
