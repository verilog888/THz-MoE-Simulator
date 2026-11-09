"""Decode a frozen gzip JSONL routing trace without extra dependencies."""
import argparse
import gzip
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace');a=p.parse_args()
    with gzip.open(a.trace,'rt',encoding='utf-8') as h:
        for line in h:sys.stdout.write(line)


if __name__=='__main__':main()
