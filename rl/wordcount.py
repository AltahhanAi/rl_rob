import json
from pathlib import Path

def count_notebook_words(nb_path=None):
    '''Count words in code and markdown cells of a Jupyter notebook.
    
    Args:
        nb_path: Path to .ipynb file. If None, uses the most recently
                 modified notebook in the current folder.
    
    Returns:
        dict with 'code', 'markdown', and 'total' word counts.
    '''
    
    if nb_path is None: 
        nb_path = max(Path('.').glob('*.ipynb'), key=lambda p: p.stat().st_mtime)
    with open(nb_path, 'r', encoding='utf-8') as f: 
        nb = json.load(f)
    def word_count(text): 
        return len(''.join(text).split())
    def output_text(o): 
        return o.get('text') or o.get('data', {}).get('text/plain') or o.get('traceback') or ''

    code = sum(word_count(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    md   = sum(word_count(c['source']) for c in nb['cells'] if c['cell_type'] == 'markdown')
    out  = sum(word_count(output_text(o)) for c in nb['cells'] if c['cell_type'] == 'code' for o in c.get('outputs', []))
    return {'code': code, 'markdown': md, 'out': out, 'total': code + md + out}

# Usage:
count_notebook_words()                    # auto-detects latest notebook
# count_notebook_words("my_notebook.ipynb") # or specify a file