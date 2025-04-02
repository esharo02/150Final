from music21 import *
import argparse
import os


def main():
    print("pen fifteen")

    # 2 options: import a file or check a file
    argparse = argparse.ArgumentParser(description="Choose leadsheet file to run")
    # argparse.add_argument("select", type=str, help="Select a file")
    # argparse.add_argument("input", type=str, help="Import a file")
    argparse.add_argument("file", type=str, help="The file to be ran")
    argparse.add_argument("mode", type=str, help="The file to be ran")
    

    args = argparse.parse_args()

    FILE = args.file
    MODE = args.mode

    ## main.py --mode select --file file.mxl
    ## main.py --mode input --file newfile.mxl

    if MODE == "select" and FILE:
        print(f"Processing selected file: {FILE}")
        
    elif MODE == "input" and FILE:
        print(f"Processing input file: {FILE}")
    
    else:
        print('Error: Invalid mode. Mode must be "select" or "input".')



if __name__ == "__main__":
    main()

