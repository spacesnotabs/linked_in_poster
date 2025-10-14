import os
from pathlib import Path
from rag.chunkers import py_chunker

if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "."

    # Walk through the directory indicated by the argument and chunk all the Python files
    output_file = "chunks_output.txt"
    for root, dirs, files in os.walk(repo):
        for file in files:
            if file.endswith(".py"):
                path = Path(os.path.join(root, file))
                print(f"Processing {path}")
                for chunk in py_chunker.py_chunks(path):
                    # output the chunk to a file
                    with open(output_file, "a") as f:
                        f.write(f"Symbol: {chunk['symbol']}\n")
                        f.write(f"Lines: {chunk['start_line']} - {chunk['end_line']}\n")
                        f.write(f"Text:\n{chunk['text']}\n")
                        f.write("="*80 + "\n")

    