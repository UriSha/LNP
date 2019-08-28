import time
import torch


class Sampler():
    def __init__(self, name, print_interval, id2w, logger):
        self.name = name
        self.id2w = id2w
        self.logger = logger
        self.last_print = time.time()
        self.print_interval = print_interval


    def sample(self, sent_y, target_x, predictions):
        if time.time() - self.last_print < self.print_interval:
            return

        self.last_print = time.time()
        

        masked_positions = {}
        for i, pos_tensor in enumerate(target_x):
            pos = pos_tensor.item()
            if pos >= len(sent_y):
                break
            masked_positions[pos] = i

        orig = []
        pred = []
        for i, word_id_tensor in enumerate(sent_y):
            word_id = word_id_tensor.item()
            if word_id == 0:
                break

            if i in masked_positions:
                pred_id = torch.max(predictions[masked_positions[i]], dim=0)[1].item()
                pred.append(f"*{self.id2w[pred_id]}*")
                orig.append(f"*{self.id2w[word_id]}*")
            else:
                word = self.id2w[word_id]
                pred.append(word)
                orig.append(word)
                

        orig_str = " ".join(orig)
        pred_str = " ".join(pred)
        self.logger.log(f"{self.name} Sample:")
        self.logger.log(f"orig: {orig_str}")
        self.logger.log(f"pred: {pred_str}")
        self.logger.log()