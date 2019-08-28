import time
import torch


class Sampler():
    def __init__(self, name, print_interval, id2w, logger):
        self.name = name
        self.id2w = id2w
        self.logger = logger
        self.last_print = time.time()
        self.print_interval = print_interval


    def sample(self, context_x, context_y, target_x, target_y, predictions):
        if time.time() - self.last_print < self.print_interval:
            return
        self.last_print = time.time()
        
        i = 0
        j = 0
        pos = 0
        orig = ""
        pred = ""
        while pos < len(context_x):
            pred_id = None
            id = context_y[i]
            if context_x[i] == pos:
                if id == 0:
                    break
                i += 1
            else:
                if j >= len(target_x):
                    self.logger.log("error")
                    return
                id = target_y[j]
                if id == 0:
                    break
                pred_id = torch.max(predictions[j], dim=0)[1]
                j += 1
            pos += 1
            if pred_id is not None:
                orig += "*" + self.id2w[int(id.item())] + "* "
                pred += "*" + self.id2w[int(pred_id.item() + 1)] + "* "
            else:
                orig += self.id2w[int(id.item())] + " "
                pred += self.id2w[int(id.item())] + " "

        self.logger.log(f"{self.name} Sample:")
        self.logger.log("orig: {}".format(orig))
        self.logger.log("pred: {}".format(pred))
        self.logger.log()