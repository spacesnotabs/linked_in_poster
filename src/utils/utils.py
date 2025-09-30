def read_file(filename: str) -> str:
    contents: str = ""
    with open(file=filename, mode="r", encoding="utf-8") as file:
        contents = file.read()
    return contents