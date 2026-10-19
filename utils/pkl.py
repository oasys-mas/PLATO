import pickle
import pprint
from pathlib import Path

def pkl_to_text(pkl_path):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} not found")

    # load pickle
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    # pretty format
    formatted = pprint.pformat(data, width=120)

    # write output .txt
    out_path = pkl_path.with_suffix(".txt")
    out_path.write_text(formatted)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pkl_to_text.py file.pkl")
        exit(1)

    pkl_to_text(sys.argv[1])
