from loader import FilesystemLoader
import os

if __name__ == "__main__":
    # The user needs to pass a root path here as an argument to the script
    if len(os.sys.argv) < 2:
        print("Usage: python main.py <root_path>")
        os.sys.exit(1)

    # if user passes in -v or --verbose, we will print out entire text of each document
    verbose = False
    if len(os.sys.argv) > 2 and os.sys.argv[2] in ("-v", "--verbose"):
        verbose = True
    
    root_path = os.sys.argv[1]
    print(f"Loading documents from: {root_path}")

    loader = FilesystemLoader(
        root=root_path,
        include=("**/*.md", "**/*.txt", "**/*.py", "**/*.html"),  # tweak as needed
        exclude=("**/.git/**", "**/node_modules/**", "**/__pycache__/**"),
    )

    for i, doc in enumerate(loader.load()):
        print(f"[{i:04}] {doc.type:<14} {doc.metadata['relpath']}  ({len(doc.text)} chars)")
        if verbose:
            print(doc.text)
            print("-" * 40)