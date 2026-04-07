class BaseScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self):
        pass

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def load_state_dict(self, init_state):
        self.scheduler.load_state_dict(init_state)

    def get_state_dict(self):
        return self.scheduler.state_dict()

    def get_lr(self):
        return self.scheduler._last_lr[0]
