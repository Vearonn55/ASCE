"""Segment local wavs or download YouTube typing audio into unlabeled clips."""

from asce.cli import main

if __name__ == "__main__":
    main(["farm", *(__import__("sys").argv[1:])])
