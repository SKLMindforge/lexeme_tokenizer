import os
from tokenizers import Tokenizer

class LexemeTokenizer:
    def __init__(self, model_path=None):
        if model_path is None:
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, "Lexeme_V3_Final_Watermarked.json")
        self.tokenizer = Tokenizer.from_file(model_path)
        
        vocab = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "<?>") for i in ids]
        return "".join(tokens).replace("Ġ", " ")

    def is_perfect(self, text):
        return text == self.decode(self.encode(text))