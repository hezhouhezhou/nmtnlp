import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define the text to tokenize
text = "This is the first line of text.\nThis is the second line of text.\nAnd this is the third line of text."

# Tokenize each line of text separately using BERT
tokens = []
for line in text.split('\n'):
    line_tokens = tokenizer.tokenize(line)
    line_tokens = ['[CLS]'] + line_tokens + ['[SEP]']
    tokens.append(line_tokens)

# pad, so that each line has the same length
max_seq_length = max(len(line) for line in tokens)
for i, line_tokens in enumerate(tokens):
    padding_length = max_seq_length - len(line_tokens)
    tokens[i] = line_tokens + ['[PAD]'] * padding_length
# Convert the tokens to input features
input_ids = []
input_mask = []
segment_ids = []

for i, token_line in enumerate(tokens):
    # Convert tokens to input ids
    input_ids_line = tokenizer.convert_tokens_to_ids(token_line)
    input_ids.append(input_ids_line)

    # Create an input mask that masks all non-padding tokens
    input_mask_line = [1] * len(input_ids_line)
    padding_length = max_seq_length - len(input_ids_line)
    input_mask_line += [0] * padding_length
    input_mask.append(input_mask_line)
    print(input_mask_line)

    # Create segment ids that are all 0's for the first sequence and 1's for the second sequence
    segment_ids_line = [0] * len(input_ids_line)
    segment_ids.append(segment_ids_line)
    print(segment_ids_line)
print(input_ids)
# Convert input features to tensors
input_ids = torch.tensor(input_ids, dtype=torch.long)
input_mask = torch.tensor(input_mask, dtype=torch.long)
segment_ids = torch.tensor(segment_ids, dtype=torch.long)

# Generate input embeddings using BERT
with torch.no_grad():
    # Use the BERT model to generate embeddings
    _, pooled_output = model(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)

# Print the shape of the embeddings
print(pooled_output.shape)