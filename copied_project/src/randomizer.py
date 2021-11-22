from string import ascii_uppercase
import random

def randomizer(L, encoding=False):
    key = list(ascii_uppercase)
    val = key.copy()
    random.seed(0)
    random.shuffle(val)
    if encoding:
        encode = dict(zip(key,val))
    else:
        encode = dict(zip(val,key))

    ls = list(L)
    mapped = [encode[i] if i in encode else i for i in ls]
    mapped = ''.join(i for i in mapped)
    return mapped
