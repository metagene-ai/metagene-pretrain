# %%
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.load("tokenizer/mgfm.model") # loads the model back from disk
# %%
stats = []
for i in range(tokenizer.vocab_size):
    stats.append((i, len(tokenizer.decode([i])), tokenizer.decode([i])))
# %%
stats.sort(key=lambda x: -x[1])
for i in range(20):
    print(stats[i][0])
    print(stats[i][1])
    print(stats[i][2])
    print('\n')
# %%
len(tokenizer.decode([4096]))
# %%
read = 'TCCGGTTCCGCGCCTGAAGCGCTTAGCCTTGCCAGTGAGGAGCAACTCGTAGGCTCATTATGCAAAAGGCACGCAGTCACCCATTACTGGGCTCCTACAGGTTGTAAGCACACGGTTTCAGGTACTAACGAAAACTCATCTCGGGGCAGGCTTCGCGCTTAGATGCTTTCAGCGCTTATCCTATCCGGGCGTAGCTACTCGGCACTGCGGCTGGCGCC'
tokenizer.encode(read)
# %%
read = 'TCCGGTTCCGCGCCTGAAGCGCTTAGCCTTGCCAGTGAGGAGCAACTCGTAGGCTCATTATGCAAAAGGCACGCAGTCACCCATTACTGGGCTCCTACAGGTTGTAAGCACACGGTTTCAGGTAC'
tokenizer.encode(read)
# %%
