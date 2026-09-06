"""Стенд v2 по схеме со слайда: индикаторы + скоры базовых моделей → логрег → калибровка (Platt, отдельное окно)
→ единый порог на 5 коридоров → политика → метрики всей системы + регулярность потока."""
from __future__ import annotations
import sys, json, time, warnings
import pathlib as _pl
HERE = _pl.Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from harness import panel, idx, splits, build_table, policy, CORRIDORS, H, TOL, PURGE, THR_MONTHS, evaluate_events, labels

# --- базовые модели, чьи скоры идут в логрег ------------------------------------
def base_model(name: str):
    name = name.split("@")[0]
    if name == "hgb": return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0)
    if name == "hgb_deep": return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.03, max_iter=300, l2_regularization=1.0, min_samples_leaf=40)
    if name == "rf": return RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.5, n_jobs=-1, random_state=0)
    if name == "et": return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.5, n_jobs=-1, random_state=0)
    if name == "knn": return make_pipeline(RobustScaler(), KNeighborsClassifier(n_neighbors=200, weights="distance"))
    if name == "nb": return make_pipeline(RobustScaler(), GaussianNB())
    if name == "hgb_reg": return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0)  # регрессия выгоды, бп
    raise ValueError(name)

def time_folds(dates: pd.Series, k: int = 5, purge: int = PURGE):
    """Блочные фолды по времени с зазором на краях: OOF-скор без утечки соседних меток."""
    qs = np.quantile(dates.astype("int64"), np.linspace(0, 1, k + 1))
    for i in range(k):
        va = (dates.astype("int64") >= qs[i]) & (dates.astype("int64") <= qs[i + 1])
        lo, hi = dates[va].min() - pd.Timedelta(days=purge * 1.5), dates[va].max() + pd.Timedelta(days=purge * 1.5)
        tr = ~((dates >= lo) & (dates <= hi))
        yield tr.to_numpy(), va.to_numpy()

def base_scores(names: list[str], tr: pd.DataFrame, others: list[pd.DataFrame], feat_cols: list[str], targets: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """OOF-скоры на обучении и скоры полных моделей на прочих таблицах. `targets[name]` — столбец цели базовой модели."""
    oof = np.zeros((len(tr), len(names))); outs = [np.zeros((len(o), len(names))) for o in others]
    Xtr = tr[feat_cols].to_numpy()
    for j, n in enumerate(names):
        tcol = targets.get(n, n.split("@")[1] if "@" in n else "y"); y = tr[tcol].to_numpy()
        ok = ~np.isnan(y)
        is_reg = n.split("@")[0].endswith("_reg")
        for trm, vam in time_folds(tr.date):
            m = base_model(n); m.fit(Xtr[trm & ok], y[trm & ok].astype(float if is_reg else int))
            oof[vam, j] = m.predict(Xtr[vam]) if is_reg else m.predict_proba(Xtr[vam])[:, 1]
        m = base_model(n); m.fit(Xtr[ok], y[ok].astype(float if is_reg else int))
        for o, arr in zip(others, outs): arr[:, j] = m.predict(o[feat_cols].to_numpy()) if is_reg else m.predict_proba(o[feat_cols].to_numpy())[:, 1]
    return oof, outs

def choose_theta(p: np.ndarray, hit: np.ndarray, base: float, mode: str, param: float, n_corr_days: float) -> float:
    """Единый порог по окну порога. fpw — квантиль под частоту; lift — максимум срабатываний при lift на окне порога ≥ param;
    utility — максимум Σ(hit − λ·(1−hit)) по отобранным дням."""
    order = np.argsort(-p); ps, hs = p[order], hit[order]
    ok = ~np.isnan(hs)
    if mode == "fpw":
        return float(np.quantile(p, 1 - min(0.5, param / 5.0)))
    if mode == "lift":
        cum_hit = np.cumsum(np.where(ok, hs, 0)) / np.maximum(1, np.cumsum(ok))
        n = np.arange(1, len(ps) + 1)
        good = (cum_hit >= param * base) & (n >= 20)
        k = np.where(good)[0].max() if good.any() else 19
        return float(ps[k])
    if mode == "utility":
        u = np.cumsum(np.where(ok, hs - param * (1 - hs), 0)); k = int(np.argmax(u)); k = max(k, 19)
        return float(ps[k])
    raise ValueError(mode)

def regularity(events_by_corr: dict[str, pd.DatetimeIndex], start, end) -> dict:
    """Регулярность потока по всем коридорам вместе: доля недель / двухнедельных периодов хотя бы с одним пушем, самый долгий перерыв."""
    all_ev = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in events_by_corr.values()])))
    weeks = pd.period_range(start, end, freq="W"); busy_w = set(all_ev.to_period("W"))
    share_w = float(np.mean([w in busy_w for w in weeks])) if len(weeks) else np.nan
    two = ((weeks - weeks[0]).map(lambda d: d.n) // 2) if len(weeks) else []
    bw = pd.Series([w in busy_w for w in weeks], index=list(two)) if len(weeks) else pd.Series(dtype=float)
    share_2w = float(bw.groupby(level=0).max().mean()) if len(bw) else np.nan
    edges = [pd.Timestamp(start), *all_ev, pd.Timestamp(end)]
    longest = max((b - a).days for a, b in zip(edges[:-1], edges[1:]))
    return {"share_weeks_any": share_w, "share_2weeks_any": share_2w, "longest_gap_any_days": float(longest), "n_any": int(len(all_ev))}

def run(spec: dict) -> dict:
    t0 = time.time()
    groups = set(spec.get("extra", []))
    tab = build_table(groups, spec.get("target", "hit_mean20"))
    # доп. цели для базовых моделей
    for c in CORRIDORS:
        r = panel[c].dropna(); m = tab["corr"] == c; d = tab.loc[m, "date"]
        tab.loc[m, "y_h10"] = labels.hit_buy_now(r, 10, TOL, "mean").reindex(d).to_numpy()
        tab.loc[m, "y_lmin"] = labels.local_min_label(r, 20, 10.0).reindex(d).to_numpy()
        tab.loc[m, "y_ben"] = labels.benefit_fwd_bps(r, 20).reindex(d).to_numpy()
    aux = ["y_h10", "y_lmin", "y_ben"]
    feat_cols = [c for c in tab.columns if c not in ("y", "corr", "date", "hit_eval", *aux)]
    fixed_cols = [c for c in feat_cols if not c.startswith(("vol", "atr", "rsi", "z2", "z6", "macd", "dd_", "ma20", "usd_", "local_", "eur", "cny", "eurusd", "usdcny", "dom", "dow", "from25"))]
    base_names = spec.get("bases", []); base_targets = spec.get("base_targets", {})
    lr_cols = fixed_cols if spec.get("lr_inputs") == "fixed" else feat_cols
    if spec.get("rank_only"):
        feat_cols = [c for c in feat_cols if c.startswith(("rank","c_","sin","cos","down_","up_","rsi","z2","z6","atr_rank","usd_rank","local_rank","eurusd_rank","from25","dow","dom"))]
        lr_cols = [c for c in lr_cols if c in feat_cols]   # что из индикаторов видит логрег напрямую
    rows = []; ev_all = {}
    for s in splits:
        thr_start = s.test_start - pd.DateOffset(months=spec.get('thr_months', THR_MONTHS)); thr_end = s.train_end
        tr_end = idx[idx.searchsorted(thr_start) - PURGE - 1]
        tr = tab[(tab.date <= tr_end) & tab.y.notna()].dropna(subset=feat_cols)
        thr = tab[(tab.date > thr_start) & (tab.date <= thr_end)].dropna(subset=feat_cols)
        te = tab[(tab.date >= s.test_start) & (tab.date <= s.test_end)].dropna(subset=feat_cols)
        w = None
        if spec.get("decay"):
            age = (tr_end - tr.date).dt.days / 365.25; w = np.power(0.5, age / spec["decay"]).to_numpy()
        # 1) базовые модели → скоры (OOF на обучении)
        if base_names:
            oof, (s_thr, s_te) = base_scores(base_names, tr, [thr, te], feat_cols, base_targets)
        else:
            oof, s_thr, s_te = np.zeros((len(tr), 0)), np.zeros((len(thr), 0)), np.zeros((len(te), 0))
        Xtr = np.c_[tr[lr_cols].to_numpy(), oof]; Xthr = np.c_[thr[lr_cols].to_numpy(), s_thr]; Xte = np.c_[te[lr_cols].to_numpy(), s_te]
        # 2) логрег-агрегатор
        lr = make_pipeline(RobustScaler(), LogisticRegression(C=spec.get("C", 0.1), max_iter=3000))
        lr.fit(Xtr, tr.y.to_numpy().astype(int), **({"logisticregression__sample_weight": w} if w is not None else {}))
        raw_thr, raw_te = lr.decision_function(Xthr), lr.decision_function(Xte)
        # 3) калибровка Platt на окне порога → вероятность «хорошего дня»
        hit_thr = thr.hit_eval.to_numpy(); okh = ~np.isnan(hit_thr)
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(raw_thr[okh].reshape(-1, 1), hit_thr[okh].astype(int))
        p_thr = platt.predict_proba(raw_thr.reshape(-1, 1))[:, 1]; p_te = platt.predict_proba(raw_te.reshape(-1, 1))[:, 1]
        # 4) единый порог на 5 коридоров
        mode, param = spec.get("thr", ["fpw", 2.0])
        gate = spec.get("gate_rank120")
        gcol = spec.get("gate_col", "rank120")
        def gm(d):
            if gcol == "any": return ((d.rank60 <= gate) | (d.rank120 <= gate)).to_numpy()
            if gcol == "fact":  # хотя бы один факт для текста: уровень, застой после снижения, провал к тренду
                level = (d.rank120 <= gate) | (d.rank60 <= gate)
                stall = (d.ret20 <= -0.01) & (d.ret5.abs() <= 0.003)
                dip = d.dip_ma60 <= -0.01
                return (level | stall | dip).to_numpy()
            return (d[gcol] <= gate).to_numpy()
        gmask = gm(thr) if gate is not None else np.ones(len(thr), bool)
        base = float(np.nanmean(hit_thr))
        if mode == "rolling":
            # единый порог на все коридоры в каждый день: квантиль калиброванного скора по хвосту последних `roll` дней публикации
            # (только прошлое: окно порога + уже прошедшие дни теста), заново каждый день — адаптируется к режиму
            roll = int(spec.get("roll_days", 120)); q = 1 - min(0.5, param / 5.0)
            allp = pd.DataFrame({"date": np.r_[thr.date.to_numpy(), te.date.to_numpy()], "p": np.r_[p_thr, p_te],
                                 "g": np.r_[gmask, gm(te.reset_index()) if gate is not None else np.ones(len(te), bool)]})
            days = np.sort(allp.date.unique()); theta_by_day = {}
            for i, d in enumerate(days):
                if d < s.test_start: continue
                lo = days[max(0, i - roll)]; hist = allp[(allp.date >= lo) & (allp.date < d) & allp.g]
                theta_by_day[d] = float(np.quantile(hist.p, q)) if len(hist) >= 50 else np.inf
            theta = float(np.median([v for v in theta_by_day.values() if np.isfinite(v)]))
        else:
            theta = choose_theta(p_thr[gmask], hit_thr[gmask], base, mode, param, len(thr) / 5)
        # 5) политика и метрики по коридорам
        ev_split = {}
        for c in CORRIDORS:
            t = te[te["corr"] == c].set_index("date"); pt = p_te[(te["corr"] == c).to_numpy()]
            sel = (pt >= np.array([theta_by_day.get(d, np.inf) for d in t.index])) if mode == "rolling" else (pt >= theta)
            if gate is not None: sel &= gm(t.reset_index())
            ev_dates = policy(t.index[sel], **spec.get("policy", {})); ev_split[c] = ev_dates
            ev_all.setdefault(c, []).extend(list(ev_dates))
            r = panel[c].dropna(); events = pd.Series(False, index=r.index); events.loc[ev_dates] = True
            m = evaluate_events(r, events, "BUY_NOW", H, (s.test_start, s.test_end), TOL, with_ci=False)
            m.update(split=s.id, corr=c, theta=theta); rows.append(m)
    df = pd.DataFrame(rows)
    if spec.get("dump"): df.to_csv(str(OUT / f"rows2_{spec['name']}.csv"), index=False)
    per_c = df.groupby("corr").agg(lift_med=("lift_mean", "median"), n=("n_events", "sum"), longest=("longest_gap_days", "max"))
    ns = df.n_scored.sum()
    reg = regularity({c: pd.DatetimeIndex(v) for c, v in ev_all.items()}, splits[0].test_start, splits[-1].test_end)
    per_c_gap = {c: float(max((b - a).days for a, b in zip([splits[0].test_start, *sorted(v), splits[-1].test_end][:-1], [splits[0].test_start, *sorted(v), splits[-1].test_end][1:]))) for c, v in ev_all.items()}
    out = {"name": spec["name"], "lift_med": float(per_c.lift_med.median()), "lift_min_corr": float(per_c.lift_med.min()),
           "lift_pooled": float(((df.hit_mean * df.n_scored).sum() / ns) / ((df.base_mean * df.n_scored).sum() / ns)),
           "share_win_gt1": float((df.lift_mean > 1).mean()), "benefit_excess": float(df.benefit_excess_bps.median()),
           "benefit_excess_pooled": float(((df.benefit_fwd_bps - df.benefit_random_day_bps) * df.n_scored).sum() / ns),
           "fpw": float(df.freq_per_week.mean()), "n_events": int(df.n_events.sum()), "clump_share": float(df.clump_share_series.mean()),
           "empty_month_share": float(df.empty_month_share.mean()), "longest_gap_corr_max": float(max(per_c_gap.values())),
           **reg, "sec": round(time.time() - t0, 1), "per_corr_lift": json.dumps({k: round(v, 3) for k, v in per_c.lift_med.items()}),
           "spec": json.dumps(spec, ensure_ascii=False)}
    return out

if __name__ == "__main__":
    specs = json.loads(open(sys.argv[1]).read()); outp = sys.argv[2]
    import os
    done = set()
    if os.path.exists(outp):
        try: done = set(pd.read_csv(outp).name)
        except Exception: pass
    for sp in specs:
        if sp["name"] in done: continue
        try: res = run(sp)
        except Exception as e:
            import traceback; res = {"name": sp["name"], "error": traceback.format_exc()[-400:], "spec": json.dumps(sp, ensure_ascii=False)}
        pd.DataFrame([res]).to_csv(outp, mode="a", header=not os.path.exists(outp), index=False)
        keys = ("lift_med", "lift_min_corr", "lift_pooled", "benefit_excess", "fpw", "n_events", "share_weeks_any", "share_2weeks_any", "longest_gap_any_days", "empty_month_share", "error", "sec")
        print(sp["name"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k in keys}, flush=True)
