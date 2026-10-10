"""Обновить палитру исходников и ноутбуков без изменения расчетов."""
from pathlib import Path
import nbformat
root=Path(__file__).resolve().parents[1]
replacement={'#263b56':'#D65A4A','#607d9c':'#E69A90','#37a59a':'#00897B',
             '#277b71':'#00897B','#d18a48':'#D65A4A','#c16b39':'#D65A4A',
             '#956d9c':'#80CBC4',"'YlGnBu'":"'Greens'","'viridis'":"'Greens'",
             "'RdBu_r'":"'BrBG'"}
palette="\nfrom cycler import cycler\nplt.rcParams['axes.prop_cycle']=cycler(color=['#D65A4A','#00897B','#E69A90','#80CBC4'])\n"
def update(text):
    for old,new in replacement.items(): text=text.replace(old,new)
    text=text.replace("color=['#D65A4A','#D65A4A']", "color=['#00897B','#D65A4A']")
    text=text.replace("color=['#D65A4A','#D65A4A','#00897B','#80CBC4']", "color=['#D65A4A','#E69A90','#00897B','#80CBC4']")
    text=text.replace("[(False,'#D65A4A'),(True,'#D65A4A')]", "[(False,'#00897B'),(True,'#D65A4A')]")
    text=text.replace("COLORS = ['#D65A4A', '#E69A90'", "COLORS = ['#00897B', '#E69A90'")
    # Diverging scales retain signed differences using green / coral.
    text=text.replace("plt.get_cmap(cmap).copy()", "(plt.get_cmap(cmap).copy() if cmap!='BrBG' else __import__('matplotlib').colors.LinearSegmentedColormap.from_list('coral_green',['#00897B','#ffffff','#D65A4A']))")
    if 'import matplotlib.pyplot as plt' in text and "axes.prop_cycle" not in text:
        text=text.replace('import matplotlib.pyplot as plt','import matplotlib.pyplot as plt'+palette)
    return text
for path in (root/'scripts').glob('build*notebook.py'):
    path.write_text(update(path.read_text(encoding='utf-8')),encoding='utf-8')
for path in (root/'notebooks').glob('*.ipynb'):
    nb=nbformat.read(path,as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code': cell.source=update(cell.source)
    nbformat.write(nb,path)
