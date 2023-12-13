import torch
from torch.utils.data import Dataset
import torchtext.transforms as transforms
import csv
from transformers import BertTokenizer


class IMDB(Dataset):

    def __init__(self, path):
        super().__init__()
        with open(path) as file:
            csvreader = csv.reader(file, delimiter=',')
            self.data = [i for i in csvreader]

        # self.label_transform = transforms.Sequential(
        #     transforms.LabelToIndex()
        # )
        # self.tz = BertTokenizer.from_pretrained('bert-base-cased')
        # self.tokenized_data = []
        # for i in self.data:
        #     self.tokenized_data += self.tz.tokenize(i[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        value = self.data[index]
        item, label = value
        item = self.text_to_tensor(item)
        return item, label

    @staticmethod
    def text_to_tensor(text):
        char_to_index = {char: idx for idx, char in enumerate(set(text))}
        text_indices = [char_to_index[char] for char in text]
        tensor = torch.tensor(text_indices, dtype=torch.long)
        return tensor
