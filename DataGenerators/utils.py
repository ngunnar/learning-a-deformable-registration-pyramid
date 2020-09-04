from pandas import DataFrame
from evalutils.io import CSVLoader

def load_pairs(fname):
    return DataFrame(CSVLoader().load(fname=fname))