class Plotter():
    def __init__(self, train_loss_per_epoch, eval_loss_per_epoch, files_timestamp, grid=True, save=True):
        self.train_loss_per_epoch = train_loss_per_epoch
        self.eval_loss_per_epoch = eval_loss_per_epoch
        self.grid = grid
        self.save = save
        self.files_timestamp = files_timestamp


    def plot(self):
        if self.save:
            with open(f"{self.files_timestamp}_train_results.txt", "w") as f:
                for l in self.train_loss_per_epoch:
                    print(f"{l}", file=f)

            with open(f"{self.files_timestamp}_eval_results.txt", "w") as f:
                for l in self.eval_loss_per_epoch:
                    print(f"{l}", file=f)

        try:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.train_loss_per_epoch)), self.train_loss_per_epoch, label="Training")
            plt.plot(range(len(self.eval_loss_per_epoch)), self.eval_loss_per_epoch, label="Evaluation")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('NLL Loss Per Epoch')
            plt.grid(self.grid)
            plt.legend()
            plt.savefig('loss_graph.png')
        except Exception:
            # fix for nova not having matplotlib installed
            print("Failed generating graph")