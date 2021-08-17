import nltk

with open('tokenz.txt') as f:
    contents = f.read()

tokens  = nltk.word_tokenize(contents)
print(tokens)
pos = nltk.pos_tag(tokens)
print(pos)

with open("Tokens_output.txt", "a") as f:
    print(tokens, file=f)

with open("Pos_output.txt", "a") as f:
    print(pos, file=f)