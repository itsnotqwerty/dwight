import tiktoken


class TiktokenWrapper:
    """Thin wrapper around a tiktoken encoding."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        return self._enc.eot_token
