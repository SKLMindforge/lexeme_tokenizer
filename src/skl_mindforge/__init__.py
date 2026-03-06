import os
from tokenizers import Tokenizer

class LexemeTokenizer:
    """
    Lexeme V3 Ultra PRO - Bit-Perfect Technical Tokenizer.
    Optimized for Science, Math, and Alchemy.
    Built by: Shreeman_Iyer | Org: SKL_Lexeme
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Dynamically locate the JSON file within the installed package
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, "Lexeme_V3_Final_Watermarked.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found at: {model_path}")
            
        self.tokenizer = Tokenizer.from_file(model_path)

    def encode(self, text):
        """Converts string to a list of token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids, skip_special_tokens=False):
        """Correctly decodes IDs back to string, handling multi-byte Unicode characters."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def is_perfect(self, text):
        """Verifies if the round-trip (Encode -> Decode) matches the input exactly."""
        return text == self.decode(self.encode(text))

    def __len__(self):
        """Returns the total vocabulary size."""
        return self.tokenizer.get_vocab_size()
