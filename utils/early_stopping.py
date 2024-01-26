class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, eps=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            eps (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.eps = eps
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = val_loss
        if self.best_score is None:
            self.best_score = score

        elif score <= self.best_score + self.eps:
            self.counter += 1

            if self.verbose:
                self.trace_func(f'[TRAINING] EarlyStopping: {self.counter}/{self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0

class YOLOEarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = float('inf')
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch
        self.early_stop = False

    def __call__(self, epoch, fitness):
        if self.best_fitness >= fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness

        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        self.early_stop = delta >= self.patience  # stop training if patience exceeded
