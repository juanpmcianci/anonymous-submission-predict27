"""Reproduce frozen empirical stages in a new directory without overwriting evidence."""
import argparse
from pathlib import Path
import empirical_evaluation as evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="code/empirical_reproduction", help="new, empty directory inside the project")
    args = parser.parse_args()
    output = (evaluation.ROOT / args.out).resolve()
    if not output.is_relative_to(evaluation.ROOT):
        parser.error("Output must be inside the project so manifests keep portable relative paths")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory")
    evaluation.OUT = output
    evaluation.prepare()
    evaluation.predict()
    evaluation.score()


if __name__ == "__main__":
    main()
