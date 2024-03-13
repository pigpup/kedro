
import pandas as pd
from typing import Dict
from transformers import AutoTokenizer
from typing import Dict, Any
from torch.utils.data import  Dataset


#def all(truthful_qa: pd.DataFrame):




def preprocess_truthful_qa(truthful_qa: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the truthful_qa dataset.

    Args:
        truthful_qa: The raw truthful_qa data in DataFrame format.

    Returns:
        A DataFrame with preprocessed 'question' and 'best_answer' columns, combined into a new 'input_text' column.
    """
    # Ensuring that the DataFrame contains the 'question' and 'best_answer' columns before proceeding
    if 'question' not in truthful_qa.columns or 'best_answer' not in truthful_qa.columns:
        raise ValueError("The DataFrame does not contain the necessary 'question' and 'best_answer' columns.")

    # Creating a new DataFrame with the 'input_text' column
    preprocessed_data = truthful_qa[['question', 'best_answer']].copy()
    preprocessed_data['input_text'] = preprocessed_data['question'] + "\n" + preprocessed_data['best_answer']

    return preprocessed_data[['input_text']]


def tokenize_text(preprocessed_data: pd.DataFrame, model_id: str, max_length: int = 256) -> Dict[str, Any]:
    """
    Tokenizes the text data and returns a dictionary containing tokenized data and other information.

    Args:
        preprocessed_data: A DataFrame containing the preprocessed text data.
        model_id: The ID of the pretrained model to use for tokenization.
        max_length: The maximum length for the tokens.

    Returns:
        A dictionary containing the tokenized text data and other related information.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    texts = preprocessed_data['input_text'].tolist()
    print (texts[0])
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')


    
    return encodings

