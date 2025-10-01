import fitz  # PyMuPDF

def read_file(filename: str) -> str:
    contents: str = ""

    if filename.lower().endswith(".pdf"):
        contents = pdf_to_text(filename)
    else:
        with open(file=filename, mode="r", encoding="utf-8") as file:
            contents = file.read()
    return contents

def pdf_to_text(filepath: str) -> str:
    text = ""
    with fitz.open(filepath) as pdf:
        for page in pdf:
            text += page.get_text()
    return text
