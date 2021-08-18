import re


def prep_seq(seq):
    """
    Adding spaces between AAs and replace rare AA [UZOB] to X.
    ref: https://huggingface.co/Rostlab/prot_bert.

    seq: a string of AA sequence.
    """
    seq_spaced = " ".join(seq)
    seq_input = re.sub(r"[UZOB]", "X", seq_spaced)
    return seq_input
