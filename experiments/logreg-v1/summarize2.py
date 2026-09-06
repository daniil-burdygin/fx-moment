iimport pathlib as _pl
HERE = _pl.Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
mport pandas as pd, glob, csv, sys
E = str(OUT) + "/"
hdr=None; rows=[]
for f in sorted(glob.glob(E+"g2_*.csv")):
    for r in csv.reader(open(f)):
        if r and r[0]=="name" and len(r)>5: hdr=r; continue
        if hdr and len(r)==len(hdr) and r[0]!="name": rows.append(r)
df=pd.DataFrame(rows,columns=hdr).drop_duplicates("name")
for c in hdr:
    if c not in ("name","spec","per_corr_lift"): df[c]=pd.to_numeric(df[c],errors="coerce")
cols=["name","lift_med","lift_min_corr","lift_pooled","share_win_gt1","benefit_excess","fpw","n_events","share_weeks_any","share_2weeks_any","longest_gap_any_days","longest_gap_corr_max","empty_month_share","sec"]
pd.set_option("display.width",300); pd.set_option("display.max_colwidth",50)
print("вариантов:",len(df))
print("\n--- топ по lift_pooled"); print(df.sort_values("lift_pooled",ascending=False)[cols].head(15).round(3).to_string(index=False))
ok=df[(df.share_2weeks_any>=0.85)&(df.longest_gap_any_days<=45)]
print("\n--- регулярные (≥85% двухнедельных периодов с пушем, перерыв ≤45 дн.), топ по lift_pooled"); print(ok.sort_values("lift_pooled",ascending=False)[cols].head(15).round(3).to_string(index=False))
ok2=df[(df.lift_pooled>=1.3)]
print("\n--- lift_pooled ≥ 1.3, топ по регулярности"); print(ok2.sort_values(["share_2weeks_any","lift_pooled"],ascending=False)[cols].head(10).round(3).to_string(index=False))
df.sort_values("lift_pooled",ascending=False).to_csv(E+"all_results_v2.csv",index=False)
