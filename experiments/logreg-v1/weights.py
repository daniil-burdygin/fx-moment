"""Веса признаков логрега по окнам walk-forward и вклад признаков в отправленные пуши (вариант: логрег + доп. индикаторы, гейт 0.3, порог 2/нед.)."""
import sys, json, numpy as np, pandas as pd, warnings
import pathlib as _pl
HERE = _pl.Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from harness import panel, idx, splits, build_table, policy, CORRIDORS, PURGE, THR_MONTHS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
ALL = {"vol","tech","usd","basket","cal"}
tab = build_table(ALL, "hit_mean20")
feat_cols = [c for c in tab.columns if c not in ("y","corr","date","hit_eval")]
fam = {}
for c in feat_cols:
    if c.startswith("c_"): fam[c] = "коридор"
    elif c.startswith("rank"): fam[c] = "уровень (ранг)"
    elif c in ("down_streak","up_streak","ret5","ret20"): fam[c] = "моментум / серия"
    elif c in ("dip_ma60","dist_min60","dd_max120","ma20_slope"): fam[c] = "отклонение от тренда"
    elif c.startswith(("sin","cos")): fam[c] = "сезонность"
    elif c in ("rsi14","z20","z60","macd"): fam[c] = "техника (RSI, z, MACD)"
    elif c.startswith(("vol","atr")): fam[c] = "волатильность"
    elif c.startswith(("usd","local")): fam[c] = "USD-контекст"
    elif c.startswith(("eur","cny","usdcny")): fam[c] = "корзина EUR/CNY"
    else: fam[c] = "календарь"
W = []; contrib_rows = []
for s in splits:
    thr_start = s.test_start - pd.DateOffset(months=THR_MONTHS); thr_end = s.train_end
    tr_end = idx[idx.searchsorted(thr_start) - PURGE - 1]
    tr = tab[(tab.date <= tr_end) & tab.y.notna()].dropna(subset=feat_cols)
    thr = tab[(tab.date > thr_start) & (tab.date <= thr_end)].dropna(subset=feat_cols)
    te = tab[(tab.date >= s.test_start) & (tab.date <= s.test_end)].dropna(subset=feat_cols)
    sc = RobustScaler().fit(tr[feat_cols]); lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(tr[feat_cols]), tr.y.astype(int))
    coef = lr.coef_[0]; W.append(pd.Series(coef, index=feat_cols, name=s.id))
    p_thr = lr.predict_proba(sc.transform(thr[feat_cols]))[:,1]; theta = float(np.quantile(p_thr, 1 - 2.0/5.0))
    Xte = sc.transform(te[feat_cols]); p_te = lr.predict_proba(Xte)[:,1]
    te = te.assign(p=p_te)
    for c in CORRIDORS:
        m = (te["corr"] == c).to_numpy(); t = te[m].set_index("date")
        sel = (t.p >= theta) & (t.rank120 <= 0.3)
        ev = policy(t.index[sel.to_numpy()])
        Xc = Xte[m][sel.to_numpy()]; Xc = Xc[[d in set(ev) for d in t.index[sel.to_numpy()]]]
        for row in Xc:
            contrib = coef * row
            contrib_rows.append({"split": s.id, "corr": c, **{f: v for f, v in zip(feat_cols, contrib)}})
W = pd.DataFrame(W).T  # признаки × окна
C = pd.DataFrame(contrib_rows)
out = {}
# 1) средний нормированный вес |coef| / Σ|coef| по окнам (без one-hot коридора)
core = [c for c in feat_cols if not c.startswith("c_")]
absn = W.loc[core].abs() / W.loc[core].abs().sum()
out["mean_norm_weight"] = absn.mean(axis=1).sort_values(ascending=False)
out["sign"] = np.sign(W.loc[core]).mean(axis=1)   # +1 стабильно положительный, −1 стабильно отрицательный
out["sign_stability"] = (np.sign(W.loc[core]).eq(np.sign(W.loc[core]).mode(axis=1)[0], axis=0)).mean(axis=1)
# 2) доля пушей, где признак — главный положительный вклад
top = C[core].idxmax(axis=1); out["share_top"] = top.value_counts(normalize=True)
# 3) средняя доля признака в положительном вкладе пуша
pos = C[core].clip(lower=0); share = pos.div(pos.sum(axis=1), axis=0); out["mean_contrib_share"] = share.mean().sort_values(ascending=False)
df = pd.DataFrame({"вес": out["mean_norm_weight"], "знак": out["sign"], "стаб.знака": out["sign_stability"], "доля_пушей_топ": out["share_top"].reindex(core).fillna(0), "доля_вклада": out["mean_contrib_share"]}).sort_values("вес", ascending=False)
df["семейство"] = [fam[c] for c in df.index]
pd.set_option("display.width", 220)
print("=== признаки, средний нормированный |вес| по 11 окнам"); print(df.round(3).head(22).to_string())
print("\n=== по семействам: сумма нормированного веса, доля пушей с топ-признаком из семейства, доля вклада")
famdf = df.groupby("семейство").agg(вес=("вес","sum"), доля_пушей_топ=("доля_пушей_топ","sum"), доля_вклада=("доля_вклада","sum")).sort_values("вес", ascending=False); print(famdf.round(3).to_string())
print("\n=== топ-8 признаков: нормированный вес по окнам (знак сохранён)")
top8 = df.index[:8]; signed = (W.loc[core] / W.loc[core].abs().sum()); lab = ["2021-I","2021-II","2022-I","2022-II","2023-I","2023-II","2024-I","2024-II","2025-I","2025-II","2026-I"]
t8 = signed.loc[top8]; t8.columns = lab; print(t8.round(3).to_string())
print("\n=== число пушей по окнам:", C.groupby("split").size().to_dict())
df.to_csv(str(OUT / "weights_summary.csv")); t8.to_csv(str(OUT / "weights_by_window.csv")); famdf.to_csv(str(OUT / "weights_families.csv"))
