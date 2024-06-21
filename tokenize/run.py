# # %%
# from tqdm import tqdm
# line_count = 0
# token_count = 0
# with open('data/MJ-2024-05-17-44_3-26_S1_L001.collapsed', 'r') as f:
#     line = f.readline()
#     while line:
#         if line[0] in ['A', 'C', 'G', 'T']:
#             line_count += 1
#             token_count += len(line.strip())
#         line = f.readline()
# # %%
# print(line_count, token_count)
# print(f'{line_count:_}', f'{token_count:_}')
# # %%
# # data/JR-2024-04-16-nR347G1-P001-L001.collapsed: 
#     # line_count: 156579515
#     # token_count: 27_075_334_829

# # data/MJ-2024-05-17-44_3-26_S1_L001.collapsed:
#     # line_count: 278328548
#     # token_count: 46822224293

# def compute_line_and_token_count(filename):
#     line_count = 0
#     token_count = 0
#     with open(filename, 'r') as f:
#         line = f.readline()
#         while line:
#             if line[0] in ['A', 'C', 'G', 'T']:
#                 line_count += 1
#                 token_count += len(line.strip())
#             line = f.readline()
#     return {
#         'line_count': line_count,
#         'token_count': token_count
#     }

# def filter_for_mg(filename, line_count = None):
#     fout = open(f'{filename.split('-')[0]}.txt', 'w')
#     pbar = tqdm(total=line_count)
#     with open(filename, 'r') as f:
#         line = f.readline()
#         while line:
#             if line[0] in ['A', 'C', 'G', 'T']:
#                 fout.write(line)
#                 pbar.update(1)
#             line = f.readline()

# filter_for_mg('data/MJ-2024-05-17-44_3-26_S1_L001.collapsed', 278328548)
# # %%
# with open('data/MJ.txt', 'r') as f:
#     for i in range(10):
#         line = f.readline()
#         print(line.strip())
#         print(len(line.strip()))
# # %%



# %%

def prepare_train_string(budget_per_file):
    # files = ['data/MJ-small.txt', 'data/JR-small.txt']

    files = ['data/cleaned_tokens_2000000000.txt']
    budget_per_file = 1_000_000_000
    # budget_per_file = 100_000_000

    train_string = []
    for file in files:
        from tqdm import tqdm
        pbar = tqdm(total=budget_per_file)
        
        with open(file, 'r') as f:
            file_train_string = [line.strip() for line in f.readlines()]
        train_string += file_train_string
    train_string = ' '.join(train_string)[:budget_per_file]
    train_string = train_string[:train_string.rfind(' ')]
    print(f'Token count: {len(train_string):_}')
    # print(f'Found {duplicates} duplicates')
    return train_string

train_string = prepare_train_string(2_500_000_000)

# %%
from minbpe import RegexTokenizer

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
tokenizer = RegexTokenizer(GPT2_SPLIT_PATTERN)
tokenizer.train(train_string, vocab_size=1024, verbose=True)
tokenizer.register_special_tokens({"<|eos|>": 1024})
tokenizer.save('tokenizer/large-mgfm-1024')
