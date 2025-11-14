from temporaldata import Data

class MaskingBase:

    def __init__(self, masked_fields: List(Tuple[str, str])):
        """
        Initialize the MaskingBase class.
        Args:
            masked_fields: A list of tuples containing the information about the fields to be masked.
            Each tuple contains the following information:
            - data_field: The name of the data field to be masked.
            - information_field: The name of the information field to be used to mask the data field.
        """
        self.masked_fields = masked_fields

    def __call__(self, data: Data) -> Data:
        mask = self._mask_fn(data)

    def check

    @abstractmethod
    def _mask_fn(self, data: Data) -> Data: 
        pass