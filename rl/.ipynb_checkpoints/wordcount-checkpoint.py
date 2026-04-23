import json
from pathlib import Path

def count_notebook_words(nb_path=None):
    """Count words in code and markdown cells of a Jupyter notebook.
    
    Args:
        nb_path: Path to .ipynb file. If None, uses the most recently
                 modified notebook in the current folder.
    
    Returns:
        dict with 'code', 'markdown', and 'total' word counts.
    """
    if nb_path is None:
        nb_path = max(Path('.').glob('*.ipynb'), key=lambda p: p.stat().st_mtime)
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    def word_count(cell):
        return len(''.join(cell['source']).split())
    
    code = sum(word_count(c) for c in nb['cells'] if c['cell_type'] == 'code')
    md   = sum(word_count(c) for c in nb['cells'] if c['cell_type'] == 'markdown')
    
    print(f"Notebook:       {Path(nb_path).name}")
    print(f"Code words:     {code}")
    print(f"Markdown words: {md}")
    print(f"Total:          {code + md}")
    
    return {'code': code, 'markdown': md, 'total': code + md}


# Usage:
# count_notebook_words()                      # auto-detects latest notebook
# count_notebook_words("my_notebook.ipynb") # or specify a file