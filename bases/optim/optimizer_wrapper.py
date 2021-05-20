class OptimizerWrapper:
    """
    A wrapper to make optimizer more concise
    """

    def __init__(self, model, optimizer, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def step(self, inputs, labels):
        self.zero_grad()
        loss = self.model.loss(inputs, labels)
        loss.backward()
        return self.optimizer.step()

    def zero_grad(self):
        self.model.zero_grad()

    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_last_lr(self):
        if self.lr_scheduler is None:
            return self.optimizer.defaults["lr"]
        else:
            return self.lr_scheduler.get_last_lr()[0]
