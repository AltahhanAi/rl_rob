'''
This file is for checking the students' local repo version 
against the frozen repo in git to ensure stability and reproducibility.

'''
import subprocess, pathlib

# def check_version(SHA):
#     repo = pathlib.Path(__file__).parent.parent
#     out = subprocess.check_output(
#         ["git","-C",str(repo),"log","-1","--oneline"], text=True).strip()
#     print("you are on the latest version" if out.startswith(SHA) else f"you are not on the frozen version: {out}")



import pathlib, subprocess

def check_version(SHA):
    if len(SHA) < 7:
        raise ValueError(f"SHA must be at least 7 characters, got {len(SHA)}")
    repo = pathlib.Path(__file__).parent.parent
    local = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if local.startswith(SHA):
        print("you are on the frozen version ✓")
    else:
        print(f"you are NOT on the frozen version (you're on {local[:7]}, expected {SHA[:7]})")