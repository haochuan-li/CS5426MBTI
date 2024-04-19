import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from peft import LoraConfig, get_peft_model 
from torch.utils.data import Dataset, DataLoader
import re
from accelerate import Accelerator
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import transformers
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import re
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

batch_size=4
data_path = './data/twitter_MBTI.csv'
output_model_path = './model/BLOOM_560M_LORA_MB_ie.pth'
add_token_length = 256
df = pd.read_csv(data_path,sep=",")
max_token_length = 1024
accelerator = Accelerator()

promptTemplate = """Base on the below twitter text, select the right the MBTI class label from the following list 
['intj',
 'intp',
 'entj',
 'entp',
 'infj',
 'infp',
 'enfj',
 'enfp',
 'istj',
 'isfj',
 'estj',
 'esfj',
 'istp',
 'isfp',
 'estp',
 'esfp']:
TEXT:<{text}>
<sep>
CLASS:<{label}>
"""
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

"""
    Dataset  
"""
def clean_label(text):
    # 去除前后空格
    text = text.strip()
    # 去除句号
    text = text.rstrip('。')
    # 去除换行符
    text = text.replace('\n', '')
    return text

def clean_text(text):
    text = re.sub(r"http\S+", "",text) #URLs
    html=re.compile(r'<.*?>')
    text = html.sub(r'',text) #html tags
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') #punctuations
    return text

def is_numerical(text):
   return isinstance(text, float) or isinstance(text, int)

class InstructionDataset(Dataset):
    def __init__(self, datafram, prompt_col, tokenizer):
        self.datafram = datafram
        self.prompt_col = prompt_col
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.datafram)

    def __getitem__(self, idx):
        prompt = self.datafram.loc[idx, self.prompt_col]
        tokenized_prompt = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=max_token_length+add_token_length)
        return tokenized_prompt

def applyPromptTemplate(row):
    text_tokens = tokenizer.encode(row['text'])
    truncated_text_tokens = text_tokens[:max_token_length]
    truncated_text = tokenizer.decode(truncated_text_tokens)
    return promptTemplate.format(text=truncated_text, label=row['label'])


df['label'] = df['label'].apply(lambda x: x[0])
df['text'] = df['text'].apply(lambda x: clean_text(str(x)))
df.dropna(inplace=True)
df = df.reset_index(drop=True)

# Initialize StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, val_indices =  [], []
for train_index, test_index in sss.split(df['text'], df['label']):
  train_indices.append(train_index)
  val_indices.append(test_index)
train_indices_flat = [idx for sublist in train_indices for idx in sublist]
val_indices_flat = [idx for sublist in val_indices for idx in sublist]


df['prompt'] = df.apply(applyPromptTemplate, axis=1)
train_df = df.loc[train_indices_flat]
train_df = train_df.reset_index(drop=True)
val_df = df.loc[val_indices_flat]
val_df = val_df.reset_index(drop=True)
print("all samples: ", len(df))
print("training samples: ", len(train_df))
print("testing samples: ", len(val_df))


"""
    Model
"""
class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, train_dataloader=None, eval_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        return self.eval_dataloader

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)

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

config = LoraConfig(
    r=16, #attention heads 
    lora_alpha=32, #alpha scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloomz-560m", #Replace from big model to small model bigscience/bloom-7b1
    load_in_8bit=True, 
    device_map={'':torch.cuda.current_device()} ,#'auto',
)

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability  
    param.data = param.data.to(torch.float32)

train_dataset = InstructionDataset(train_df, 'prompt', tokenizer)
val_dataset = InstructionDataset(val_df, 'prompt', tokenizer)


# Accelerate封装模型
model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()
model.lm_head = CastOutputToFloat(model.lm_head)
model = get_peft_model(model, config)
print_trainable_parameters(model)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        pin_memory=True,
    )
val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        pin_memory=True,
    )

model, train_dataloader, val_dataloader = accelerator.prepare(
    model, train_dataloader, val_dataloader
)

trainer = CustomTrainer(
        model=model,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=1000,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            output_dir='outputs',
        ),
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
    )

model.module.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

def extract_cast_output_to_float(model):
    for name, module in model.named_children():
        if isinstance(module, CastOutputToFloat):
            setattr(model, name, module[0])  # 将 CastOutputToFloat 模块替换为其子模块
        else:
            extract_cast_output_to_float(module)  # 递归处理子模块
            
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
extract_cast_output_to_float(unwrapped_model)
accelerator.save(unwrapped_model.state_dict(), output_model_path)