import nltk
import jieba
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer

# read corpus
def read_corpus(is_chn, source):
    # 打开文件并读取内容
    if not is_chn:
        with open(source, 'r', encoding='utf-8') as f:
            data = f.read()
        en_tokens = []
        # 对英文进行分词
        a = data.split('\n')
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        tokenized_dataset = [[token.lower() for token in tokenizer.tokenize(sentence)] for sentence in a]
        vocabulary_en = list(word for sentence in tokenized_dataset for word in sentence)
        return tokenized_dataset, vocabulary_en

    else: 
        # # 对中文进行分词
        with open(source, 'r', encoding='utf-8') as f:
            data = f.read()
        zh_tokens = []
        b = data.split('\n')

        tokenized_dataset_zh = [list(jieba.cut(sentence)) for sentence in b]
        vocabulary_zh = list(word for sentence in tokenized_dataset_zh for word in sentence)
        return tokenized_dataset_zh, vocabulary_zh


def get_embeddings_bert(is_chn,source):
    model = BertModel.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    if not is_chn:
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    with open(source, 'r', encoding='utf-8') as f:
        d = f.read()
    
    tokens = []
    max1 = 0
    for line in d.split('\n')[:10000]:
        line_tokens = tokenizer.tokenize(line)
        line_tokens = ['[CLS]'] + line_tokens + ['[SEP]']
        if(len(line_tokens) > max1):
            print(len(line_tokens))
            max1 = len(line_tokens)
            print(line_tokens)
        if(len(line_tokens) >= 512):
            continue
        tokens.append(line_tokens)

    # Pad the tokens to ensure that all lines have the same length
    max_seq_length = max(len(line) for line in tokens)
    print(max_seq_length)
    for i, line_tokens in enumerate(tokens):
        padding_length = max_seq_length - len(line_tokens)
        tokens[i] = line_tokens + ['[PAD]'] * padding_length

    #Convert the tokens to input features
    input_ids = []
    input_mask = []
    segment_ids = []
    for i, token_line in enumerate(tokens):
        print(i)
        # only process first 10000 tokens
        if i >= 10000:
            break
    
        # Convert tokens to input ids
        input_ids_line = tokenizer.convert_tokens_to_ids(token_line)
        input_ids.append(input_ids_line)
        if len(token_line) > 512:
            print(len(token_line),token_line,i)
            break

        # Create an input mask that masks all non-padding tokens
        input_mask_line = [1] * len(input_ids_line)
        padding_length = max_seq_length - len(input_ids_line)
        input_mask_line += [0] * padding_length
        input_mask.append(input_mask_line)

        # Create segment ids that are all 0's for the first sequence and 1's for the second sequence
        segment_ids_line = [0] * len(input_ids_line)
        segment_ids.append(segment_ids_line)
    # Convert input features to tensors
    print(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    print("converting to tensors")
    # Generate input embeddings using BERT
    with torch.no_grad():
        # Use the BERT model to generate embeddings
        sb, pooled_output = model(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
    return sb, pooled_output

if __name__ == "__main__":
    # en_tokenized_data_sets,en_vocab = read_corpus(False,"sample.en")
    # zh_tokenized_data_sets,zh_vocab = read_corpus(True,"sample.zh")
    en_encoded_layers, en = get_embeddings_bert(False, "/Users/hezhou/projects/nmtNLP/sample.en")
    print("english words done")
    zh_encoded_layers, zh = get_embeddings_bert(True, "/Users/hezhou/projects/nmtNLP/sample.zh")
    print("chinese words done")

    #zh_encoded_layers, zh_tokens,zh_indexed_tokens = get_embeddings_bert(True, "/Users/hezhou/projects/nmtNLP/news-commentary-v16.zh")
    print(en_encoded_layers.size())
    

   # print(sb.size(),pooled_out.size())
    #print(zh_encoded_layers[2].size(),len(zh_tokens),len(zh_indexed_tokens))



