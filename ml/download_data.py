import sys
import os
import urllib.request
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset import filter_mate_puzzles

def download_file(url: str, filename: str):
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = downloaded * 100 / total_size
            sys.stdout.write(f"\rDownloading... {percent:.2f}%")
            sys.stdout.flush()

    print(f"Downloading {url} to {filename}")
    urllib.request.urlretrieve(url, filename, reporthook)
    print()

def decompress_zst(input_file: str, output_file: str):
    print(f"Decompressing {input_file} to {output_file}")
    try:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(input_file, 'rb') as ifh, open(output_file, 'wb') as ofh:
            dctx.copy_stream(ifh, ofh)
    except ImportError:
        print("zstandard not found, trying subprocess with zstd")
        subprocess.run(['zstd', '-d', input_file, '-o', output_file], check=True)

if __name__ == '__main__':
    url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zst_filename = str(data_dir / "lichess_db_puzzle.csv.zst")
    csv_filename = str(data_dir / "lichess_db_puzzle.csv")
    filtered_filename = str(data_dir / "mate_puzzles.csv")

    if not os.path.exists(zst_filename) and not os.path.exists(csv_filename):
        download_file(url, zst_filename)
        
    if not os.path.exists(csv_filename):
        decompress_zst(zst_filename, csv_filename)
        
    print(f"Filtering mate puzzles into {filtered_filename}...")
    num_puzzles = filter_mate_puzzles(csv_filename, filtered_filename)
    print(f"Filtering complete! Kept {num_puzzles} puzzles.")
