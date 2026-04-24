'''
This file is for checking the students' local repo version 
against the frozen repo in git to ensure stability and reproducibility.

'''
import subprocess, pathlib

def check_version(SHA):
    repo = pathlib.Path(__file__).parent.parent
    out = subprocess.check_output(
        ["git","-C",str(repo),"log","-1","--oneline"], text=True).strip()
    print("you are on the latest version" if out.startswith(SHA) else f"you are not on the frozen version: {out}")
