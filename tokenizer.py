"""
[개념 1] 토크나이저 — 텍스트를 숫자로, 숫자를 텍스트로.

LLM은 글자를 직접 다루지 못한다. 오직 숫자(정수 인덱스)만 다룬다.
그래서 가장 먼저 필요한 것이 "글자 <-> 숫자" 사전이다.

여기서는 가장 단순한 '글자 단위(character-level)' 토크나이저를 쓴다.
  - 코퍼스에 등장하는 모든 서로 다른 글자를 모아서 vocabulary(어휘)를 만든다.
  - 각 글자에 0, 1, 2, ... 정수를 부여한다.
  - "안녕" -> [12, 87] 처럼 인코딩하고, 다시 [12, 87] -> "안녕" 으로 디코딩한다.

진짜 GPT는 글자가 아니라 'BPE(subword)' 토크나이저를 쓴다. 자주 붙어 다니는
글자 덩어리("_the", "ing" 등)를 하나의 토큰으로 묶어 시퀀스를 짧게 만든다.
원리는 같다 — 결국 텍스트를 정수 시퀀스로 바꾸는 사전일 뿐이다.
글자 단위는 vocab이 작고 구현이 단순해서 '이론을 이해하기'에 가장 좋다.

--- English ---
[Concept 1] Tokenizer — text to numbers, numbers to text.

An LLM cannot deal with characters directly. It only handles numbers (integer indices).
So the very first thing we need is a "character <-> number" dictionary.

Here we use the simplest 'character-level' tokenizer.
  - Collect every distinct character that appears in the corpus to build a vocabulary.
  - Assign each character an integer 0, 1, 2, ...
  - Encode "안녕" -> [12, 87], and decode [12, 87] -> "안녕" back again.

A real GPT uses a 'BPE (subword)' tokenizer rather than characters. It groups
frequently co-occurring chunks of characters ("_the", "ing", etc.) into a single
token to make sequences shorter. The principle is the same — it is ultimately just a
dictionary that turns text into an integer sequence. Character-level has a small vocab
and a simple implementation, which makes it the best choice for 'understanding the theory'.
"""

from pathlib import Path


class CharTokenizer:
    def __init__(self, text: str):
        # 코퍼스에 나온 모든 고유 글자를 정렬해서 어휘로 삼는다.
        # Sort all unique characters found in the corpus and use them as the vocabulary.
        # (정렬하는 이유: 실행할 때마다 같은 글자에 같은 번호가 붙도록 — 재현성)
        # (why sort: so the same character gets the same index on every run — reproducibility)
        chars = sorted(set(text))
        self.chars = chars
        self.vocab_size = len(chars)
        # stoi: string -> integer  (글자 -> 번호)
        # stoi: string -> integer  (character -> index)
        # itos: integer -> string  (번호 -> 글자)
        # itos: integer -> string  (index -> character)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        # 어휘에 없는 글자는 건너뛴다 (코퍼스에 없는 글자가 들어와도 KeyError 안 남)
        # Skip characters not in the vocabulary (no KeyError even for out-of-corpus chars)
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_corpus(path: str = "corpus_kor.txt") -> str:
    # __file__ 기준으로 찾아서, 어느 폴더에서 실행해도 코퍼스를 읽을 수 있게 한다.
    # Resolve the path relative to __file__ so the corpus can be read no matter which folder you run from.
    p = Path(__file__).parent / path
    return p.read_text(encoding="utf-8")


if __name__ == "__main__":
    # 이 파일을 직접 실행하면 토크나이저가 뭘 하는지 눈으로 보여준다.
    # Running this file directly shows visually what the tokenizer does.
    text = load_corpus()
    tok = CharTokenizer(text)
    print(f"코퍼스 길이(글자 수): {len(text)}")
    print(f"어휘 크기(고유 글자 수, vocab_size): {tok.vocab_size}")
    print(f"어휘 앞부분: {tok.chars[:30]}")
    sample = "아침에 눈을 뜨면"
    ids = tok.encode(sample)
    print(f"\n인코딩:  {sample!r}\n     -> {ids}")
    print(f"디코딩:  {ids}\n     -> {tok.decode(ids)!r}")
