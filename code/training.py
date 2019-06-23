import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class Trainer():
    def __init__(self, model, to_cuda):
        self.model = model
        self.to_cuda = to_cuda

    def train(self):
        loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)

        train_loss_history = []
        test_recalls = []
        best_recall = float('-inf')
        for epoch in range(epoch_count):
        epoch_train_loss = []
        epoch_train_recall = []
        
        # train
        train_model(model)
        for target_xs_batch, target_ys_batch in train_loader:
            target_xs = torch.stack(target_xs_batch, dim=1).float()
            target_ys = torch.stack(target_ys_batch, dim=1).float()
            if to_cuda:
            target_xs = target_xs.cuda()
            target_ys = target_ys.cuda()
            
            target_xs = Variable(target_xs, requires_grad=True)
            target_ys = Variable(target_ys, requires_grad=True)

            # feedforward - backprop
            optimizer.zero_grad()
            outputs = model(full_context, target_xs)
            loss = loss_function(outputs, target_ys)
            loss.backward()
            optimizer.step()

            # train loss
            epoch_train_loss.append(loss.item())
            
            # train recall
            train_recall = compute_recall(outputs, target_xs, target_ys)
            epoch_train_recall.append(train_recall)
            
        # test
        eval_model(model)

        outputs = model.decoder(model.concat_repr_to_target(test_xs))

        # compute epoch recall
        recall = compute_recall(outputs, test_xs, test_ys)
        
        # save best representation
        if recall > best_recall:
            best_recall = recall
            export_model(model, model.representation, attn, epoch)
        
        test_recalls.append(recall)

        # compute epoch loss
        cur_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        train_loss_history.append(cur_train_loss)
        
        # compute epoch average train recall
        cur_train_recall = sum(epoch_train_recall) / len(epoch_train_recall)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print('Epoch [%d/%d] Train Loss: %.4f, Train Recall: %.4f, Test Recall: %.4f' %(epoch+1, epoch_count, cur_train_loss, cur_train_recall, recall))
            print()
