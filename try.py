from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
tokens = tokenizer.tokenize('[PAD]')
print(tokens)
x = tokenizer.convert_tokens_to_ids(tokens)
print(x)