import matplotlib.pyplot as plt


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
        plt.clf()
        plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('{} NLL Loss Per Epoch'.format(prefix))
        plt.grid(self.grid)
        if self.save:
            plt.savefig('{}_loss.png'.format(prefix.lower()))
        else:
            plt.show()