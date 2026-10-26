"""Secondary-head IOI workflow, reusing the common intervention engine."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--head',choices=['9-6','10-0','both'],default='both')
    args=parser.parse_args()
    heads=[(9,6),(10,0)] if args.head=='both' else [tuple(map(int,args.head.split('-')))]
    for layer,head in heads:
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_test.py')),
                        '--split','test','--layer',str(layer),'--head',str(head)],check=True)


if __name__=='__main__':main()
