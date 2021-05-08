from sbclassifier.tokenizer import add_sparse_bigrams
from sbclassifier.tokenizer import tokenize_text


def test_add_sparse_bigrams():

    assert list(add_sparse_bigrams("a b c d e".split())) == [
        "a",
        "a_b",
        "a__c",
        "b",
        "b_c",
        "b__d",
        "c",
        "c_d",
        "c__e",
        "d",
        "d_e",
    ]


def test_it_normalizes_unicode():

    # ñ in two equivalent unicode representations
    n1 = "\u00f1"
    n2 = "\u006e\u0303"

    assert list(tokenize_text(n1)) == list(tokenize_text(n2))


def test_it_restricts_charset():
    s = "aàα"
    assert list(tokenize_text(s, restrict_charset=None)) == ["aàα"]
    assert list(tokenize_text(s, restrict_charset='latin1')) == ["aà?"]
    assert list(tokenize_text(s, restrict_charset='ascii')) == ["a??"]
