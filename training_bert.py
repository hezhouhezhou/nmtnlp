import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn.utils.rnn import pad_sequence

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
device = torch.device("mps")
model.to(device)

# Define the masked language modeling task
def mltask(input_ids, input_mask):
    # Randomly mask some tokens in the input sequence
    masked_input_ids = input_ids.clone().to(device)
    masked_input_ids[input_mask == 1] = tokenizer.mask_token_id

    # Feed the masked input sequence to BERT
    outputs = model(masked_input_ids, attention_mask=input_mask.to(device))
    hidden_states = outputs[0]

    # Project the encoder hidden states to the same dimension as the input embeddings
    projection_layer = torch.nn.Linear(hidden_states.shape[-1], tokenizer.vocab_size)
    projection_layer.to(device)
    projected_hidden_states = projection_layer(hidden_states)

    # Apply a softmax activation function to obtain the probability distribution over the vocabulary
    softmax_layer = torch.nn.Softmax(dim=-1)
    logits = softmax_layer(projected_hidden_states)

    return logits

# Prepare your input data
with open('/Users/hezhou/projects/nmtNLP/news-commentary-v16.en', 'r', encoding='utf-8') as f:
    d = f.read()
tokens = []
max1 = 0
for line in d.split('\n')[:10000]:
    line_tokens = tokenizer.tokenize(line)
    line_tokens = ['[CLS]'] + line_tokens + ['[SEP]']
    if(len(line_tokens) > max1):
        max1 = len(line_tokens)
    if(len(line_tokens) >= 512):
        continue
    tokens.append(line_tokens)

for i, line_tokens in enumerate(tokens):
    padding_length = max1 - len(line_tokens)
    tokens[i] = line_tokens + ['[PAD]'] * padding_length
to_ids = [tokenizer.convert_tokens_to_ids(line) for line in tokens]
input_ids = torch.tensor(to_ids).to(device)
input_mask = torch.ones_like(input_ids).to(device)

# divide the inputs into minibatches
batch_size = 30
num_batches = len(input_ids) // batch_size
input_ids_batches = torch.split(input_ids[:num_batches*batch_size], batch_size)
input_mask_batches = torch.split(input_mask[:num_batches*batch_size], batch_size)

# Define the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for i in range(num_batches):
    print(i)
    logits = mltask(input_ids_batches[i], input_mask_batches[i])
    non_padding = input_mask_batches[i].view(-1) == 1
    loss = loss_function(logits.view(-1, logits.shape[-1])[non_padding], input_ids_batches[i].view(-1)[non_padding])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Tokenize the input text
text = "I want to go to the ____"
tokenized_text = tokenizer.tokenize(text)

# Convert the tokenized text into input IDs and input mask
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
input_ids = torch.tensor([indexed_tokens]).to(device)
input_mask = torch.ones_like(input_ids).to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=input_mask)
    logits = outputs[0]

# Apply a softmax activation function to the logits to obtain the probability distribution over the vocabulary
softmax_layer = torch.nn.Softmax(dim=-1)
probs = softmax_layer(logits)

# Select the word with the highest probability as the predicted word
predicted_index = torch.argmax(probs[0, -1]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# Print the predicted word
print(predicted_token)
