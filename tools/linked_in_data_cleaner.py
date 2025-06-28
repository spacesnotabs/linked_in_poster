"""
Cleans data from LinkedIn files for using as LLM finetuning
"""
import pandas as pd

# files specific to LinkedIn
shared_csv = "../data/Shares.csv"
rich_media_csv = "../data/Rich_Media.csv"
comments_csv = "../data/Comments.csv"

def clean_csv(filename: str, cols: list | None = None) -> pd.DataFrame | None:
    ret = None
    try:
        # Sometimes, the export of data from LinkedIn includes some lines
        # that do not parse correctly (e.g. text broken into multiple columns).
        df = pd.read_csv(filename, on_bad_lines="warn")
        if cols is not None:
            ret = df[cols]
        else:
            ret = df
    except FileNotFoundError:
        print("Could not find file ", filename)
    
    return ret

if __name__ == "__main__":
    # Clean data from various LinkedIn export files
    df = clean_csv(filename=shared_csv, cols=["ShareCommentary"])
    df = clean_csv(filename=rich_media_csv, cols=["Media Description"])
    df = clean_csv(filename=comments_csv, cols=["Message"])