from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

device = 'cuda:0'


data = load_dataset('sade-adrien/redpajama_v2_sample_1M', split='train')

tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-base-en-v1.5', device_map=device, trust_remote_code=True)
model = AutoModel.from_pretrained('Alibaba-NLP/gte-base-en-v1.5', device_map=device, trust_remote_code=True, torch_dtype=torch.float16)


chunk_size = 1000
batch_size = 256

for i in tqdm(range(0, len(data), batch_size)):
    chunks = data[i:i+batch_size]['raw_content'][:chunk_size]
    encoded_inputs = tokenizer(chunks, truncation=True, padding=True, max_length=1024, return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = model(**encoded_inputs)

    sentence_embeddings = model_output.last_hidden_state[:, 0]
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
   
    with open(f'embeddings1M_{str(i // 100_000)}.txt','ab') as file:
        np.savetxt(file, sentence_embeddings.cpu(), delimiter=' ', fmt='%.16f')