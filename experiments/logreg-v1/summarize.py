import pandas as pd, glob, numpy as np
E = str(OUT) + "/"
import csv
import pathlib as _pl
HERE = _pl.Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
hdr = open(E+"grid2.csv").readline().strip().split(",")
rows=[]
for f in glob.glob(E+"grid*.csv"):
    for r in csv.reader(open(f)):
        if len(r)==len(hdr) and r[0]!="name": rows.append(r)
df = pd.DataFrame(rows, columns=hdr)
for c in hdr:
    if c not in ("name","spec","per_corr_lift"): df[c]=pd.to_numeric(df[c], errors="coerce")
df = df.drop_duplicates("name")
print("всего вариантов:", len(df))
cols=["name","lift_med_corr_median","lift_med_corr_min","lift_pooled","share_windows_lift_gt1","benefit_excess_med","fpw","n_events","clump_share","empty_month_share","longest_gap","per_corr_lift"]
pd.set_option("display.width",260); pd.set_option("display.max_colwidth",70)
print(df.sort_values("lift_pooled",ascending=False)[cols].head(14).round(3).to_string(index=False))
print("\n--- ориентиры и спека")
print(df[df.name.str.startswith(("ref_","spec_","lr_hit_mean20","lr_base_gate"))][cols].round(3).to_string(index=False))
df.sort_values("lift_pooled",ascending=False).to_csv(E+"all_results.csv",index=False)
b=pd.read_csv(E+"rows_BEST_lr_exAll_gate0.2_fpw2_dump.csv"); c=pd.read_csv(E+"rows_ref_calendar25_dump.csv"); l=pd.read_csv(E+"rows_ref_level_only_dump.csv"); s=pd.read_csv(E+"rows_spec_lr_base_dump.csv")
def win(d): return d.groupby("split").apply(lambda g: pd.Series({"lift":(g.hit_mean*g.n_scored).sum()/((g.base_mean*g.n_scored).sum()) if g.n_scored.sum() else np.nan,"n":g.n_events.sum()}))
W=pd.concat({"best":win(b),"calendar":win(c),"level":win(l),"spec_lr":win(s)},axis=1)
print("\n--- lift по окнам (pooled по коридорам)"); print(W.round(2).to_string())
m=b.merge(c,on=["split","corr"],suffixes=("_b","_c")); d=(m.lift_mean_b-m.lift_mean_c).dropna()
rng=np.random.default_rng(0); boots=[np.median(rng.choice(d,len(d))) for _ in range(2000)]
print(f"\nпары коридор×окно: {len(d)}, best>=calendar в {int((d>=0).sum())} из {len(d)}, медиана разности lift {d.median():.3f}, 95% CI [{np.percentile(boots,2.5):.3f}, {np.percentile(boots,97.5):.3f}]")
d2=(m.benefit_excess_bps_b-m.benefit_excess_bps_c).dropna(); boots=[np.mean(rng.choice(d2,len(d2))) for _ in range(2000)]
print(f"разность выгоды сверх случайного дня: среднее {d2.mean():.0f} бп, 95% CI [{np.percentile(boots,2.5):.0f}, {np.percentile(boots,97.5):.0f}]")
m2=b.merge(s,on=["split","corr"],suffixes=("_b","_s")); d3=(m2.lift_mean_b-m2.lift_mean_s).dropna(); boots=[np.median(rng.choice(d3,len(d3))) for _ in range(2000)]
print(f"best vs логрег по спеке: best>= в {int((d3>=0).sum())} из {len(d3)}, медиана разности {d3.median():.3f}, 95% CI [{np.percentile(boots,2.5):.3f}, {np.percentile(boots,97.5):.3f}]")
print("\n--- лучший по коридорам"); print(b.groupby("corr").apply(lambda g: pd.Series({"lift_med":g.lift_mean.median(),"share_gt1":(g.lift_mean>1).mean(),"n":g.n_events.sum(),"fpw":g.freq_per_week.mean(),"ben":g.benefit_excess_bps.median()})).round(3).to_string())

# модель против правила «уровень» при той же частоте
for bname, lname in [("BEST_lr_exAll_gate0.2_fpw2_dump","ref_level_only_fpw0.35_dump"),("best_gate0.15_fpw2_dump","ref_level_only_fpw0.25_dump")]:
    bb=pd.read_csv(E+f"rows_{bname}.csv"); ll=pd.read_csv(E+f"rows_{lname}.csv")
    mm=bb.merge(ll,on=["split","corr"],suffixes=("_b","_l")); dd=(mm.lift_mean_b-mm.lift_mean_l).dropna()
    boots=[np.median(rng.choice(dd,len(dd))) for _ in range(2000)]
    d2=(mm.benefit_excess_bps_b-mm.benefit_excess_bps_l).dropna(); boots2=[np.mean(rng.choice(d2,len(d2))) for _ in range(2000)]
    print(f"\n{bname} vs {lname}: модель>=уровень в {int((dd>=0).sum())} из {len(dd)} пар, медиана разности lift {dd.median():.3f} CI [{np.percentile(boots,2.5):.3f},{np.percentile(boots,97.5):.3f}]; выгода +{d2.mean():.0f} бп CI [{np.percentile(boots2,2.5):.0f},{np.percentile(boots2,97.5):.0f}]")
