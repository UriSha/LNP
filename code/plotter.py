import matplotlib.pyplot as plt


class Plotter():
    def __init__(self, train_loss_per_epoch, eval_loss_per_epoch):
        self.train_loss_per_epoch = train_loss_per_epoch
        self.eval_loss_per_epoch = eval_loss_per_epoch

    def plot_train(self):
        plt.plot(range(len(self.train_loss_history)), self.train_loss_history)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training Cross Entropy Loss Per Epoch')
        plt.grid(True)
        plt.show()

    def plot_eval(self):
        plt.plot(range(len(self.eval_loss_per_epoch)), self.eval_loss_per_epoch)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Evaluation Cross Entropy Loss Per Epoch')
        plt.grid(True)
        plt.show()