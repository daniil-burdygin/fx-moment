"""Экспериментальный стенд по мотивам V1-спеки: pooled-модель на индикаторах → порог → политика → метрики всей системы."""
from __future__ import annotations
import sys, json, time, itertools, warnings
import pathlib as _pl
HERE = _pl.Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from fxmoment.data.store import load_panel
from fxmoment.backtest.walkforward import make_splits
from fxmoment.metrics import evaluate_events
from fxmoment import labels
from fxmoment.indicators.base import rolling_pct_rank, down_streak, up_streak
from fxmoment.config import CORRIDORS

H, TOL = 20, 25.0
PURGE = 20
THR_MONTHS = 12

panel = load_panel()
idx = panel.index
splits = make_splits(idx)

# --- признаки -------------------------------------------------------------
def base_feats(r: pd.Series, usd: pd.Series) -> pd.DataFrame:
    """Зафиксированные для текстов индикаторы: уровень, застой, моментум, сезон, провал к тренду."""
    f = pd.DataFrame(index=r.index)
    f["rank120"] = rolling_pct_rank(r, 120)          # level
    f["rank60"] = rolling_pct_rank(r, 60)
    f["rank250"] = rolling_pct_rank(r, 250)
    f["down_streak"] = down_streak(r).astype(float)  # momentum / stall
    f["up_streak"] = up_streak(r).astype(float)
    f["ret5"] = r.pct_change(5); f["ret20"] = r.pct_change(20)
    f["dip_ma60"] = r / r.rolling(60).mean() - 1     # dip_vs_trend
    f["dist_min60"] = r / r.rolling(60).min() - 1
    m = r.index.month
    for k in (1, 2):                                  # сезонность как гармоники месяца
        f[f"sin{k}"] = np.sin(2 * np.pi * k * m / 12); f[f"cos{k}"] = np.cos(2 * np.pi * k * m / 12)
    return f

def extra_feats(r: pd.Series, usd: pd.Series, eur: pd.Series, cny: pd.Series, groups: set[str]) -> pd.DataFrame:
    f = pd.DataFrame(index=r.index)
    ret1 = r.pct_change()
    if "vol" in groups:
        f["vol20"] = ret1.rolling(20).std(); f["vol60"] = ret1.rolling(60).std()
        f["vol_ratio"] = f["vol20"] / f["vol60"]
        f["atr_rank"] = rolling_pct_rank(ret1.abs().rolling(10).mean(), 120)
    if "tech" in groups:
        d = r.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
        f["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        f["z20"] = (r - r.rolling(20).mean()) / r.rolling(20).std()
        f["z60"] = (r - r.rolling(60).mean()) / r.rolling(60).std()
        f["macd"] = (r.ewm(span=12).mean() - r.ewm(span=26).mean()) / r
        f["dd_max120"] = r / r.rolling(120).max() - 1
        f["ma20_slope"] = r.rolling(20).mean().pct_change(5)
    if "usd" in groups:
        f["usd_ret5"] = usd.pct_change(5); f["usd_ret20"] = usd.pct_change(20)
        f["usd_rank120"] = rolling_pct_rank(usd, 120); f["usd_vol20"] = usd.pct_change().rolling(20).std()
        loc = r / usd
        f["local_ret5"] = loc.pct_change(5); f["local_ret20"] = loc.pct_change(20)
        f["local_rank120"] = rolling_pct_rank(loc, 120); f["local_z60"] = (loc - loc.rolling(60).mean()) / loc.rolling(60).std()
    if "basket" in groups:
        f["eur_ret5"] = eur.pct_change(5); f["cny_ret5"] = cny.pct_change(5)
        f["eurusd_ret5"] = (eur / usd).pct_change(5); f["eurusd_rank120"] = rolling_pct_rank(eur / usd, 120)
        f["usdcny_ret5"] = (usd / cny).pct_change(5)
    if "cal" in groups:
        f["dom"] = r.index.day.astype(float); f["dow"] = r.index.dayofweek.astype(float)
        f["from25"] = (r.index.day >= 25).astype(float)
    return f

def target(r: pd.Series, kind: str) -> pd.Series:
    if kind == "hit_mean20": return labels.hit_buy_now(r, 20, TOL, "mean")
    if kind == "hit_mean10": return labels.hit_buy_now(r, 10, TOL, "mean")
    if kind == "hit_min20": return labels.hit_buy_now(r, 20, TOL, "min")
    if kind == "localmin20": return labels.local_min_label(r, 20, 10.0)
    if kind == "benefit_pos": b = labels.benefit_fwd_bps(r, 20); return (b > 0).astype(float).where(b.notna())
    if kind == "benefit_top":  # верхний терцль выгоды в скользящем прошлом — «хороший день»
        b = labels.benefit_fwd_bps(r, 20); q = b.rolling(500, min_periods=200).quantile(0.67).shift(0)
        return (b >= q).astype(float).where(b.notna() & q.notna())
    raise ValueError(kind)

def build_table(groups: set[str], tkind: str) -> pd.DataFrame:
    frames = []
    for c in CORRIDORS:
        r = panel[c].dropna()
        usd, eur, cny = panel["USD"].reindex(r.index), panel["EUR"].reindex(r.index), panel["CNY"].reindex(r.index)
        f = pd.concat([base_feats(r, usd), extra_feats(r, usd, eur, cny, groups)], axis=1)
        for cc in CORRIDORS: f[f"c_{cc}"] = float(cc == c)
        f["y"] = target(r, tkind); f["corr"] = c; f["date"] = r.index
        f["hit_eval"] = labels.hit_buy_now(r, H, TOL, "mean")
        frames.append(f)
    return pd.concat(frames, ignore_index=True)

# --- модели --------------------------------------------------------------
def make_model(name: str, C: float = 0.1, cw=None):
    if name == "logreg": return make_pipeline(RobustScaler(), LogisticRegression(C=C, max_iter=2000, class_weight=cw))
    if name == "hgb": return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, class_weight=cw)
    if name == "rf": return RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.5, n_jobs=-1, random_state=0, class_weight=cw)
    raise ValueError(name)

def fit_predict(spec: dict, Xtr, ytr, wtr, Xthr, Xte):
    kind = spec["model"]
    def one(name):
        m = make_model(name, spec.get("C", 0.1), spec.get("cw"))
        m.fit(Xtr, ytr, **({"logisticregression__sample_weight": wtr} if name == "logreg" and wtr is not None else ({"sample_weight": wtr} if wtr is not None else {})))
        return m.predict_proba(Xthr)[:, 1], m.predict_proba(Xte)[:, 1]
    if kind in ("logreg", "hgb", "rf"): return one(kind)
    if kind == "blend_lr_hgb":  # среднее рангов вероятностей
        a1, a2 = one("logreg"); b1, b2 = one("hgb")
        return (a1 + b1) / 2, (a2 + b2) / 2
    if kind == "blend_lr_rf":
        a1, a2 = one("logreg"); b1, b2 = one("rf")
        return (a1 + b1) / 2, (a2 + b2) / 2
    if kind == "stack_hgb_lr":  # OOF-скор HGB как признак логрега (стек по спеке: модели на входе логрега)
        from sklearn.model_selection import KFold
        oof = np.zeros(len(Xtr)); kf = KFold(5, shuffle=False)
        for tr, va in kf.split(Xtr):
            m = make_model("hgb"); m.fit(Xtr[tr], ytr[tr]); oof[va] = m.predict_proba(Xtr[va])[:, 1]
        m = make_model("hgb"); m.fit(Xtr, ytr)
        s_thr, s_te = m.predict_proba(Xthr)[:, 1], m.predict_proba(Xte)[:, 1]
        lr = make_model("logreg", spec.get("C", 0.1)); lr.fit(np.c_[Xtr, oof], ytr)
        return lr.predict_proba(np.c_[Xthr, s_thr])[:, 1], lr.predict_proba(np.c_[Xte, s_te])[:, 1]
    raise ValueError(kind)

# --- политика ----------------------------------------------------------
def policy(dates: pd.DatetimeIndex, cooldown: int = 3, cap: int = 2, span: int = 7) -> pd.DatetimeIndex:
    out = []
    for d in dates:
        if out and (d - out[-1]).days < cooldown: continue
        if sum(1 for x in out if (d - x).days < span) >= cap: continue
        out.append(d)
    return pd.DatetimeIndex(out)

def regularity(events_by_corr, start, end):
    all_ev = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in events_by_corr.values()])))
    weeks = pd.period_range(start, end, freq="W"); busy = set(all_ev.to_period("W"))
    flags = np.array([w in busy for w in weeks])
    share_w = float(flags.mean()); share_2w = float(np.mean([flags[i:i+2].any() for i in range(0, len(flags), 2)]))
    share_4w = float(np.mean([flags[i:i+4].any() for i in range(0, len(flags), 4)]))
    edges = [pd.Timestamp(start), *all_ev, pd.Timestamp(end)]
    gaps = sorted(((b - a).days for a, b in zip(edges[:-1], edges[1:])), reverse=True)
    return {"share_weeks_any": share_w, "share_2weeks_any": share_2w, "share_4weeks_any": share_4w, "longest_gap_any": float(gaps[0]), "gaps_over_30": int(sum(g > 30 for g in gaps)), "gaps_over_60": int(sum(g > 60 for g in gaps))}

def run(spec: dict) -> dict:
    t0 = time.time(); ev_all = {}
    groups = set(spec.get("extra", []))
    tab = build_table(groups, spec.get("target", "hit_mean20"))
    feat_cols = [c for c in tab.columns if c not in ("y", "corr", "date", "hit_eval")]
    rows = []
    for s in splits:
        thr_start = s.test_start - pd.DateOffset(months=THR_MONTHS)
        thr_end = s.train_end                                        # окно порога заканчивается зазором до теста
        tr_end_pos = idx.searchsorted(thr_start) - PURGE - 1          # обучение → зазор → окно порога
        tr_end = idx[tr_end_pos]
        tr = tab[(tab.date <= tr_end) & tab.y.notna()].dropna(subset=feat_cols)
        thr = tab[(tab.date > thr_start) & (tab.date <= thr_end)].dropna(subset=feat_cols)
        te = tab[(tab.date >= s.test_start) & (tab.date <= s.test_end)].dropna(subset=feat_cols)
        if spec.get("recent_years"): tr = tr[tr.date >= tr_end - pd.DateOffset(years=spec["recent_years"])]
        w = None
        if spec.get("decay"):  # экспоненциальный вес свежих наблюдений, полураспад в годах
            age = (tr_end - tr.date).dt.days / 365.25; w = np.power(0.5, age / spec["decay"]).to_numpy()
        if spec.get("drop_shock"):
            keep = ~((tr.date >= "2022-02-24") & (tr.date <= "2022-07-31")); tr = tr[keep]; w = w[keep.to_numpy()] if w is not None else None
        Xtr, ytr = tr[feat_cols].to_numpy(), tr.y.to_numpy().astype(int)
        if spec["model"] == "calendar":
            def cal(d):
                d = pd.DataFrame({"date": d.date.to_numpy(), "corr": d["corr"].to_numpy()}); d["ym"] = d.date.dt.to_period("M")
                first = d[d.date.dt.day >= 25].groupby(["corr", "ym"]).date.min()
                key = pd.MultiIndex.from_arrays([d["corr"], d.ym]); d["first"] = first.reindex(key).to_numpy()
                return (d.date == d["first"]).astype(float).to_numpy()
            p_thr, p_te = cal(thr), cal(te)
        elif spec["model"] == "level_only":
            p_thr, p_te = (1 - thr.rank120).to_numpy(), (1 - te.rank120).to_numpy()
        else:
            p_thr, p_te = fit_predict(spec, Xtr, ytr, w, thr[feat_cols].to_numpy(), te[feat_cols].to_numpy())
        thr = thr.assign(p=p_thr); te = te.assign(p=p_te)
        # порог: один на все коридоры, по окну порога, на целевую частоту до политики
        target_fpw = spec.get("target_fpw", 1.0)
        n_days_thr = thr.groupby("corr").size().mean()
        q = 1 - min(0.5, target_fpw * n_days_thr / 5.0 / n_days_thr * 7 / 7 * (7 / 7))  # доля дней = fpw/5 (5 дней публикации в неделю)
        q = 1 - min(0.5, target_fpw / 5.0)
        theta = 0.5 if spec["model"] == "calendar" else float(np.quantile(thr.p, q))
        gate = spec.get("gate_rank120")
        for c in CORRIDORS:
            t = te[te["corr"] == c].set_index("date")
            sel = t.p >= theta
            if gate is not None: sel &= t.rank120 <= gate
            ev_dates = policy(t.index[sel.to_numpy()], **spec.get("policy", {})); ev_all.setdefault(c, []).extend(list(ev_dates))
            r = panel[c].dropna()
            events = pd.Series(False, index=r.index); events.loc[ev_dates] = True
            m = evaluate_events(r, events, "BUY_NOW", H, (s.test_start, s.test_end), TOL, with_ci=False)
            m.update(split=s.id, corr=c, theta=theta); rows.append(m)
    df = pd.DataFrame(rows)
    if spec.get("dump_events"): json.dump({c: [d.strftime("%Y-%m-%d") for d in sorted(v)] for c, v in ev_all.items()}, open(str(OUT / f"events_{spec['name']}.json"), "w"))
    if spec.get("dump"): df.to_csv(str(OUT / f"rows_{spec['name']}.csv"), index=False)
    # сводка: медиана по окнам lift_mean на коридор, затем медиана/среднее по коридорам; pooled lift по событиям
    per_c = df.groupby("corr").agg(lift_med=("lift_mean", "median"), ben_med=("benefit_excess_bps", "median"),
                                   fpw=("freq_per_week", "mean"), n=("n_events", "sum"))
    pooled_hit = (df.hit_mean * df.n_scored).sum() / df.n_scored.sum()
    pooled_base = (df.base_mean * df.n_scored).sum() / df.n_scored.sum()
    out = {"name": spec["name"], "lift_med_corr_median": float(per_c.lift_med.median()), "lift_med_corr_min": float(per_c.lift_med.min()),
           "lift_pooled": float(pooled_hit / pooled_base), "lift_win_median_all": float(df.lift_mean.median()),
           "share_windows_lift_gt1": float((df.lift_mean > 1).mean()), "benefit_excess_med": float(df.benefit_excess_bps.median()),
           "fpw": float(df.freq_per_week.mean()), "n_events": int(df.n_events.sum()), "clump_share": float(df.clump_share_series.mean()),
           "empty_month_share": float(df.empty_month_share.mean()), "longest_gap": float(df.longest_gap_days.median()),
           "sec": round(time.time() - t0, 1), "spec": json.dumps(spec, ensure_ascii=False)}
    out["per_corr_lift"] = json.dumps({k: round(v, 3) for k, v in per_c.lift_med.items()})
    out.update(regularity({c: pd.DatetimeIndex(v) for c, v in ev_all.items()}, splits[0].test_start, splits[-1].test_end))
    out["longest_gap_corr_max"] = float(max(max((b - a).days for a, b in zip([splits[0].test_start, *sorted(v), splits[-1].test_end][:-1], [splits[0].test_start, *sorted(v), splits[-1].test_end][1:])) for v in ev_all.values()))
    return out

if __name__ == "__main__":
    specs = json.loads(open(sys.argv[1]).read()); outp = sys.argv[2]
    done = set()
    try: done = set(pd.read_csv(outp).name)
    except Exception: pass
    for sp in specs:
        if sp["name"] in done: continue
        try: res = run(sp)
        except Exception as e: res = {"name": sp["name"], "error": repr(e)[:200], "spec": json.dumps(sp, ensure_ascii=False)}
        pd.DataFrame([res]).to_csv(outp, mode="a", header=not pd.io.common.file_exists(outp), index=False)
        print(sp["name"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k in ("lift_med_corr_median", "lift_pooled", "benefit_excess_med", "fpw", "n_events", "error", "sec")}, flush=True)
