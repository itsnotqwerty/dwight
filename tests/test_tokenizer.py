"""Tests for TiktokenWrapper."""

from dwight.tokenizer import TiktokenWrapper


def test_encode_returns_list_of_ints():
    tok = TiktokenWrapper()
    ids = tok.encode("Hello, world!")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_decode_roundtrip():
    tok = TiktokenWrapper()
    text = "The quick brown fox"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_vocab_size():
    tok = TiktokenWrapper()
    assert tok.vocab_size == 100_277


def test_eot_token_is_int():
    tok = TiktokenWrapper()
    assert isinstance(tok.eot_token, int)


def test_empty_string():
    tok = TiktokenWrapper()
    ids = tok.encode("")
    assert ids == []
    assert tok.decode([]) == ""
