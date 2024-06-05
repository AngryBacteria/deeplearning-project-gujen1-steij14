from nltk.corpus import wordnet as wn

current_synset = wn.synsets('bear')[0]


def get_all_hyponyms(synset, depth=0):
    hyponyms = []
    indent = " " * (depth * 2)
    for hyponym in synset.hyponyms():
        offset = hyponym.offset()
        wnid = "n{:08d}".format(offset)
        hyponyms.append(wnid)
        hyponyms.extend(get_all_hyponyms(hyponym, depth + 1))
        print(f"{indent}{hyponym.lemma_names()} ({wnid})")

    return hyponyms


h = get_all_hyponyms(current_synset)
print(h)
