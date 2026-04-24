import subprocess
def check_version(SHA):
    out = subprocess.check_output(["git","-C",".","log","-1","--oneline"], text=True).strip()
    print("you are on the latest version" if out.startswith(SHA) else f"you are not on the forzen version: {out}")

