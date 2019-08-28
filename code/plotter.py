import os


class Plotter():
    def __init__(self, train_loss_per_epoch, eval_losses_per_epoch, tags, logger, grid=True, save=True):
        self.train_loss_per_epoch = train_loss_per_epoch
        self.eval_losses_per_epoch = eval_losses_per_epoch
        self.tags = tags
        self.grid = grid
        self.save = save
        self.log_dir = logger.log_dir


    def plot(self):
        if self.save:
            with open(os.path.join(self.log_dir, "train_results.txt"), "w") as f:
                for l in self.train_loss_per_epoch:
                    print(f"{l}", file=f)
            
            for i in range(len(self.eval_losses_per_epoch)):
                with open(os.path.join(self.log_dir, f"eval_results({self.tags[i]}).txt"), "w") as f:
                    for l in self.eval_losses_per_epoch[i]:
                        print(f"{l}", file=f)

        try:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.train_loss_per_epoch)), self.train_loss_per_epoch, label="Training")
            for i in range(len(self.eval_losses_per_epoch)):
                plt.plot(range(len(self.eval_losses_per_epoch[i])), self.eval_losses_per_epoch[i], label=f"Evaluation({self.tags[i]})")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("NLL Loss Per Epoch")
            plt.grid(self.grid)
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, "loss_graph.png"))
        except Exception:
            # fix for nova not having matplotlib installed
            print("Failed generating graph")