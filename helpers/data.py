# Description: methods to read a file

import pandas as pd

def read_file(file_path: str, encoding: str = 'ISO-8859-1') -> object:
    return pd.read_csv(file_path, encoding=encoding)

class DataReader:
    def __init__(self,file_path : str):
        self.file_path = file_path
        self.data = read_file(file_path)

    def read(self) -> object: return self.data

    def get(self, column_name : str) -> list: 
        return self.data[column_name]

    def to_list(self, column_name : str) -> list:
       #tipar lista
       return self.get(column_name).tolist() 

    def replace_values(self, replace: object, column_name: str) -> list:
        return list(map(lambda x: replace[x], self.to_list(column_name)))