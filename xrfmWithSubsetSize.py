from xrfm import xRFM


class xRFMWithSubsetSize(xRFM):
    """xRFM subclass that allows configuring the split model's AGOP subset size."""
    def __init__(self, *args, split_subset_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_subset_size = split_subset_size

    def _get_agop_on_subset(self, X, y, subset_size=50_000, **kwargs):
        if self.split_subset_size is not None:
            subset_size = self.split_subset_size
        return super()._get_agop_on_subset(X, y, subset_size=subset_size, **kwargs)