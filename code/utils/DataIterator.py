import pandas as pd

from DGADataset_ysx import DGATrueDataset_ysx


class DataIterator:
    def __init__(self, file_path, chunksize):
        self.file_path = file_path
        self.chunksize = chunksize
        self.reader = pd.read_csv(self.file_path, chunksize=self.chunksize, iterator=True)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.reader)
            return DGATrueDataset_ysx(chunk, True)
        except StopIteration:
            raise StopIteration
