from temporaldata import Data


class AddConfig:
    def __init__(self, config):
        self.config = config

    def __call__(self, data: Data) -> Data:
        data.config = self.config
        return data
