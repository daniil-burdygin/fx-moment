import sys
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject
D=sys.argv[1]
v6=PdfReader(f"{D}/Презентация_финальная_v6.pdf"); s=PdfReader(f"{D}/Слайды_защита_v7.pdf")
w=PdfWriter()
for pg in [v6.pages[0],v6.pages[1]]+list(s.pages):
    w.add_page(pg)
for pg in w.pages:
    if "/Annots" in pg: del pg[NameObject("/Annots")]
w.add_metadata({"/Title":"Момент — колода защиты v7 (команда 9)","/Author":"Даниил Бурдыгин, Илья Фазлов, Олег Андреев"})
w.write(f"{D}/Презентация_защита_v7.pdf")
print("pages:",len(w.pages))
