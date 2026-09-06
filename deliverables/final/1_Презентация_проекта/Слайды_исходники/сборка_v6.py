"""Сборка v6: v5 без 16 страниц концепта, четыре слайда прототипа, слайд «Дальнейшие шаги» из v10.
Аргумент: metrics=v5|v10 (какой слайд метрик брать)."""
import sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject
D = Path.home() / "projects/fx-moment/deliverables/final/1_Презентация_проекта"
metrics = (sys.argv[1] if len(sys.argv) > 1 else "metrics=v5").split("=")[1]
v5 = PdfReader(str(D / "Презентация_финальная_v5.pdf"))
pr = PdfReader(str(D / "Слайды_прототип.pdf"))
v10 = PdfReader(str(D / "Слайды_исходники/Презентация_v10_напарников.pdf"))
tx = PdfReader(str(D / "Слайды_исходники/Слайды_тексты_пушей.pdf"))  # стр. 2 — «устаревание» с кадром прототипа
order = [(v5, 1), (v5, 2), (v5, 3),
         (pr, 1), (pr, 2), (pr, 3), (pr, 4),
         (v5, 20), (v5, 21), ((v10, 20) if metrics == "v10" else (v5, 22)),
         (v5, 23), (v5, 24), (tx, 2), (v5, 26), (v5, 27), (v5, 28), (v5, 29),
         (v10, 26), (v5, 30), (v5, 31)]
w = PdfWriter()
for rd, n in order:
    pg = rd.pages[n - 1]
    if "/Annots" in pg: del pg[NameObject("/Annots")]   # ссылок между страницами в v6 нет
    w.add_page(pg)
out = D / "Презентация_финальная_v6.pdf"
w.write(str(out)); print("pages", len(w.pages), "metrics", metrics, "->", out.name)
