from pprint import pprint
from pathlib import Path

from code.pipeline.main import run_pipeline

IMAGE_PATH = Path("results/showcase/00a2e3ca-5c856cde.jpg")

def main():
    assert IMAGE_PATH.exists(), f"Image not found: {IMAGE_PATH}"

    output = run_pipeline(IMAGE_PATH)
    print("\n=== PIPELINE OUTPUT ===")
    pprint(output)

if __name__ == "__main__":
    main()
