import os


def main():
    with open("tiny_shakespeare.txt", "r", encoding="utf-8") as input:
        text = input.read()

    chars = sorted(list(set(text)))
    V = len(chars)

    os.makedirs("data", exist_ok=True)
    with open("data/vocab.txt", "w", encoding="utf-8") as output:
        output.write(f"{str(V)}\n")
        for ch in chars:
            match ch:
                case "\n":
                    output.write("\\n\n")
                case "\t":
                    output.write("\\t\n")
                case _:
                    output.write(f"{ch}\n")


if __name__ == "__main__":
    main()
