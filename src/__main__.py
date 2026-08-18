from llm_sdk import Small_LLM_Model


def main() -> None:
    print("Chargement du modèle...")

    model = Small_LLM_Model()

    print("Modèle chargé !")

    text = "Hello"

    encoded = model.encode(text)
    print("Texte :", text)
    print("Encoded :", encoded)

    input_ids = encoded[0].tolist()
    print("Input IDs :", input_ids)

    logits = model.get_logits_from_input_ids(input_ids)
    print("Nombre de logits :", len(logits))

    next_token_id = max(
        range(len(logits)),
        key=lambda i: logits[i]
    )

    print("Next token ID :", next_token_id)
    print("Logit :", logits[next_token_id])

    next_token = model.decode([next_token_id])
    print("Next token :", repr(next_token))


if __name__ == "__main__":
    main()
