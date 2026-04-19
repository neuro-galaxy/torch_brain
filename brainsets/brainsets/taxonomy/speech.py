from enum import IntEnum, auto
import re

import numpy as np


class CVSyllable(IntEnum):
    empty = 0
    baa = auto()
    bee = auto()
    boo = auto()
    chaa = auto()
    chee = auto()
    choo = auto()
    daa = auto()
    dee = auto()
    dhaa = auto()
    dhee = auto()
    dhoo = auto()
    doo = auto()
    faa = auto()
    fee = auto()
    foo = auto()
    gaa = auto()
    gee = auto()
    goo = auto()
    ha = auto()
    haa = auto()
    hee = auto()
    hoo = auto()
    jee = auto()
    kaa = auto()
    kay = auto()
    kee = auto()
    koo = auto()
    laa = auto()
    lee = auto()
    loo = auto()
    maa = auto()
    mee = auto()
    moo = auto()
    naa = auto()
    nee = auto()
    noo = auto()
    paa = auto()
    pee = auto()
    poo = auto()
    raa = auto()
    ree = auto()
    roo = auto()
    saa = auto()
    see = auto()
    shaa = auto()
    she = auto()
    shee = auto()
    shoo = auto()
    skee = auto()
    soo = auto()
    taa = auto()
    tee = auto()
    thaa = auto()
    thee = auto()
    thoo = auto()
    too = auto()
    vaa = auto()
    vee = auto()
    voo = auto()
    waa = auto()
    wee = auto()
    woo = auto()
    ya = auto()
    yaa = auto()
    yee = auto()
    yoo = auto()
    zaa = auto()
    zee = auto()
    zoo = auto()


# From ARPABET
class Phoneme(IntEnum):
    BLANK = 0
    AA = auto()
    AE = auto()
    AH = auto()
    AO = auto()
    AW = auto()
    AY = auto()
    B = auto()
    CH = auto()
    D = auto()
    DH = auto()
    EH = auto()
    ER = auto()
    EY = auto()
    F = auto()
    G = auto()
    HH = auto()
    IH = auto()
    IY = auto()
    JH = auto()
    K = auto()
    L = auto()
    M = auto()
    N = auto()
    NG = auto()
    OW = auto()
    OY = auto()
    P = auto()
    R = auto()
    S = auto()
    SH = auto()
    T = auto()
    TH = auto()
    UH = auto()
    UW = auto()
    V = auto()
    W = auto()
    Y = auto()
    Z = auto()
    ZH = auto()
    SIL = auto()


g2p = None

vocab = "abcdefghijklmnopqrstuvwxyz'- "
# We write phonemes using this pseudo-cypher to calculate phoneme distances
pseudo_code = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN "


def to_phonemes(sentence, add_interword_symbol=True):
    global g2p
    sentence = "".join(
        [x for x in sentence.lower().strip().replace("--", "") if x in vocab]
    )
    if g2p is None:
        from g2p_en import G2p

        # Grapheme to phoneme library
        g2p = G2p()

    sentence = g2p(sentence)
    phonemes = []
    for p in sentence:
        if add_interword_symbol and p == " ":
            phonemes.append("SIL")
        p = re.sub(r"[0-9]", "", p)  # Remove stress
        if re.match(r"[A-Z]+", p):  # Only keep phonemes
            phonemes.append(p)

    # add one SIL symbol at the end so there's one at the end of each word
    if add_interword_symbol:
        phonemes.append("SIL")

    sentence = np.array([int(Phoneme[x]) for x in phonemes])
    pseudo_sentence = "".join([pseudo_code[x] for x in sentence])
    return pseudo_sentence, sentence


def from_phonemes(sentence):
    # Use the greedy decoding approach, without a language model.
    last_c = -1
    pseudo_transcription = []
    transcription = []
    for c in sentence:
        if c == last_c:
            continue
        last_c = c
        if c == 0:
            continue
        pseudo_transcription.append(pseudo_code[c])
        transcription.append(Phoneme(c).name)
    return ("".join(pseudo_transcription)), (
        ("·".join(transcription)).replace("·SIL·", " ").replace("·SIL", "")
    )
