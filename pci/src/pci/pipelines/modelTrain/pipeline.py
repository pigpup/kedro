

from typing import Dict, Any



from kedro.pipeline import Pipeline, node
from .nodes import  prepare_model, train_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=prepare_model,
                inputs={ "model_id": "params:tokenize_model_id","train_dataset" : "tokenized_TrainingData"},
                outputs="prepared_model",
                name="preparing_model",
            ),
      
            node(
                func=train_model,
                inputs={
                    "model": "prepared_model", 
                    "train_dataset": "tokenized_TrainingData",
                    "model_id": "params:tokenize_model_id"
                }, 
                outputs="trained_model",
                name="training_model",
            ),
        ]
    )
