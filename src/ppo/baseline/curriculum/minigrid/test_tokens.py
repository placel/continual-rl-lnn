unique_words = [
    'get', 'a', 'go', 'fetch', 'you', 'must', 'to', 'the', 'matching',
    'object', 'at', 'end', 'of', 'hallway', 'traverse', 'rooms', 'goal',
    'put', 'near', 'and', 'open', 'red', 'door', 'then', 'blue', 'pick',
    'up', 'green', 'grey', 'purple', 'yellow', 'box', 'key', 'ball', 'square',
    'use', 'it', 'next', 'first', 'second', 'third'
]

PAD_IDX = 0
UNK_IDX = 1
# Start vocab indexing from 2 to leave 0 for PAD and 1 for UNK
word_dict = {word: idx + 2 for idx, word in enumerate(unique_words)}

# Build reverse dictionary
idx_to_word = {idx: word for word, idx in word_dict.items()}
idx_to_word[PAD_IDX] = "<PAD>"
idx_to_word[UNK_IDX] = "<UNK>"

tokens = [ 5,  3, 26, 35,  0,  0,  0,  0,  0,  0,  0,  0]

for t in tokens:
    print(idx_to_word.get(t, "<UNK>"))