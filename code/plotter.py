class Plotter():
    def __init__(self, train_loss_per_epoch, eval_loss_per_epoch, grid=True, save=True):
        self.train_loss_per_epoch = train_loss_per_epoch
        self.eval_loss_per_epoch = eval_loss_per_epoch
        self.grid = grid
        self.save = save


    def plot_train(self):
        self.plot_loss('Training', self.train_loss_per_epoch)


    def plot_eval(self):
        if self.eval_loss_per_epoch:
            self.plot_loss('Evaluation', self.eval_loss_per_epoch)


    def plot_loss(self, prefix, loss_per_epoch):
        plt.plot(range(len(loss_per_epoch)), loss_per_epoch, label=prefix)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('{} NLL Loss Per Epoch'.format(prefix))
        plt.grid(self.grid)
        if self.save:
            plt.savefig('{}_loss.png'.format(prefix.lower()))
        else:
            plt.show()

    def plot(self):
        if self.save:
            with open("train_results.txt", "w") as f:
                for l in self.train_loss_per_epoch:
                    print(f"{l}", file=f)

            with open("eval_results.txt", "w") as f:
                for l in self.eval_loss_per_epoch:
                    print(f"{l}", file=f)

        import matplotlib.pyplot as plt
        plt.plot(range(len(self.train_loss_per_epoch)), self.train_loss_per_epoch, label="Training")
        plt.plot(range(len(self.eval_loss_per_epoch)), self.eval_loss_per_epoch, label="Evaluation")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('NLL Loss Per Epoch')
        plt.grid(self.grid)
        plt.legend()
        plt.savefig('loss_graph.png')