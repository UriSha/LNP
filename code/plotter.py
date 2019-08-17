import os


class Plotter():
    def __init__(self, train_loss_per_epoch, eval_loss_per_epoch, log_dir, grid=True, save=True):
        self.train_loss_per_epoch = train_loss_per_epoch
        self.eval_loss_per_epoch = eval_loss_per_epoch
        self.grid = grid
        self.save = save
        self.log_dir = log_dir


    def plot(self):
        if self.save:
            with open(os.path.join(self.log_dir, "train_results.txt"), "w") as f:
                for l in self.train_loss_per_epoch:
                    print(f"{l}", file=f)

            with open(os.path.join(self.log_dir, "eval_results.txt"), "w") as f:
                for l in self.eval_loss_per_epoch:
                    print(f"{l}", file=f)

        try:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.train_loss_per_epoch)), self.train_loss_per_epoch, label="Training")
            plt.plot(range(len(self.eval_loss_per_epoch)), self.eval_loss_per_epoch, label="Evaluation")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("NLL Loss Per Epoch")
            plt.grid(self.grid)
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, "loss_graph.png"))
        except Exception:
            # fix for nova not having matplotlib installed
            print("Failed generating graph")