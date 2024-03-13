import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

import pandas as pd
import datasets 
print(torch.__version__)  


def prepare_model(model_id: str, train_dataset):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    
    print_trainable_parameters(model)

    
    print(torch.__version__)  
    print(train_dataset) 
    train_dataset = TextDataset(train_dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    training_args = transformers.TrainingArguments(
        num_train_epochs=30,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        lr_scheduler_type='cosine'
    )
    

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    model.config.use_cache = False
   
   
    trainer.train()

    
    return model

    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """


    
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def generate( tokenizer, model):

  example_text = 'can i eact cats'
  correct_answer = 'maybe if you are a dog'

  print("Question:")
  print(example_text)
  model.to('cuda:0')
  encoding = tokenizer(example_text, return_tensors="pt").to("cuda:0")
  output = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask, max_new_tokens=100, do_sample=True, temperature=0.000001, eos_token_id=tokenizer.eos_token_id, top_k = 0)

  print("Answer:")
  print(tokenizer.decode(output[0], skip_special_tokens=True))

  print("Best Answer:")
  print(correct_answer)

  print()


def train_model(model, train_dataset,  model_id: str):
    
    return 
    print(torch.__version__)  
    print(train_dataset) 
    
    train_dataset = TextDataset(train_dataset)
    print(train_dataset[0]) 
     
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    generate(tokenizer,model)
    return 
    


class TextDataset(  torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])