from kedro.pipeline import Pipeline, node
from typing import Dict, Any
from .nodes import preprocess_truthful_qa,  tokenize_text

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_truthful_qa,
                inputs="truthful_qa",
                outputs="preprocessed_truthful_qa",
                name="preprocessing_truthful_qa",
            ),
node(
    func=tokenize_text,
    inputs={
        "preprocessed_data": "preprocessed_truthful_qa", 
        "model_id": "params:tokenize_model_id"
    },
    outputs="tokenized_TrainingData",
    name="tokenizing_truthful_qa",
)
        ]
    )