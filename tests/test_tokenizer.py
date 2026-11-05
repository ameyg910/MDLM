from maskforge.data.tokenizer import MaskForgeTokenizer


def test_tokenizer_training_and_encoding():

    texts = [
        "The cat sat on the mat.",
        "The dog sat on the floor.",
        "Diffusion models generate text.",
        "Transformers process token sequences.",
    ] * 100

    tokenizer = MaskForgeTokenizer.train(
        texts,
        vocab_size=500,
    )

    encoded = tokenizer.encode(
        "The cat sat on the mat."
    )

    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id

    assert all(
        0 <= token_id < tokenizer.vocab_size
        for token_id in encoded
    )

    decoded = tokenizer.decode(encoded)

    assert decoded == "The cat sat on the mat."


def test_tokenizer_save_and_load(tmp_path):

    texts = [
        "Hello world.",
        "Diffusion language models.",
    ] * 100

    tokenizer = MaskForgeTokenizer.train(
        texts,
        vocab_size=300,
    )

    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)

    loaded = MaskForgeTokenizer.load(path)

    original = tokenizer.encode("Hello world.")
    restored = loaded.encode("Hello world.")

    assert original == restored
    assert loaded.vocab_size == tokenizer.vocab_size