"""Gym member segments: who should get the monthly $10 extras credit?

Reproduces every number on the slides. Run:  python analysis.py
Outputs go to ./outputs/

Steps (match the assignment):
 1. 70/30 split (seed 42); scaler fit on training members only
 2. k-means for k = 1..8 on training members; elbow on within-group spread
 3. test spread: test members assigned to training centers (no refit)
 4. decision function: segment + credit yes/no for one member; applied to the new member
 5. value: monthly net from the credit, member by member, baseline vs v1 (all 330 members)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
K_RANGE = range(1, 9)
K_CHOSEN = 4
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# member_id is an identifier, not a trait, so it is left out.
FEATURES = ["age", "visits_per_month", "avg_visit_minutes",
            "classes_per_month", "months_member", "monthly_extras_spend"]

# Finance's working assumptions (per member per month)
CREDIT_COST = 10          # every member who gets the credit uses it
SPEND_THRESHOLD = 20      # >= $20 on extras: would spend the same anyway
LIFT_SPEND = 25           # < $20: spends $25 more with the credit
MARGIN = 0.60             # margin on extras


def credit_net(spend):
    """Gym's monthly net from giving one member the credit (member-level rule)."""
    spend = np.asarray(spend, dtype=float)
    return np.where(spend >= SPEND_THRESHOLD, -CREDIT_COST, LIFT_SPEND * MARGIN - CREDIT_COST)


def spread(X, centers):
    """Sum of squared distances from each member to its nearest center."""
    d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return float(d.min(axis=1).sum())


# ---------- 1. data and split ----------
df = pd.read_csv("data/gym_members.csv")
assert df.shape == (330, 7)
train, test = train_test_split(df, test_size=0.30, random_state=SEED)
scaler = StandardScaler().fit(train[FEATURES])           # fit on training only
Xtr, Xte = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])

# ---------- 2-3. elbow: training and test spread ----------
models, rows = {}, []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xtr)
    models[k] = km
    rows.append({"k": k, "train_spread": km.inertia_, "test_spread": spread(Xte, km.cluster_centers_)})
elbow = pd.DataFrame(rows)
for c in ["train_spread", "test_spread"]:
    elbow[c.replace("spread", "pct_of_k1")] = 100 * elbow[c] / elbow[c].iloc[0]
elbow.round(1).to_csv(OUT / "elbow_table.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(elbow.k, elbow.train_spread, "o-", color="black", label=f"Training members (n={len(train)})")
ax.plot(elbow.k, elbow.test_spread, "s--", color="gray", label=f"Test members (n={len(test)})")
ax.axvline(K_CHOSEN, color="black", lw=0.8, ls=":")
ax.set_xlabel("Number of segments (k)")
ax.set_ylabel("Within-group spread\n(sum of squared distances)")
ax.set_xticks(list(K_RANGE))
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "elbow_train_test.png", dpi=200); plt.close()

# ---------- chosen model ----------
km = models[K_CHOSEN]
# relabel clusters 0..k-1 by average extras spend (low -> high) so labels are stable
raw_centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATURES)
order = raw_centers["monthly_extras_spend"].sort_values().index.tolist()
relabel = {old: new for new, old in enumerate(order)}

# Segment names: chosen from the centroids (see segment_profile.csv)
SEGMENT_NAMES = {0: "Newcomers", 1: "Loyal Veterans", 2: "Solo Grinders", 3: "Class Devotees"}


def segment_of(features_df):
    raw = km.predict(scaler.transform(features_df[FEATURES]))
    return np.array([relabel[c] for c in raw])


df["segment"] = segment_of(df)
df["set"] = np.where(df.index.isin(train.index), "train", "test")
df["credit_net_if_given"] = credit_net(df.monthly_extras_spend)

# ---------- 4. decision: which segments get the credit ----------
# Decide on TRAINING members only: credit a segment if giving it the credit nets > $0.
train_seg = df[df.set == "train"].groupby("segment")["credit_net_if_given"].sum()
CREDIT_SEGMENTS = sorted(int(s) for s in train_seg[train_seg > 0].index)


def decide(member: dict):
    """One member's features -> (segment name, gets credit?)"""
    seg = int(segment_of(pd.DataFrame([member]))[0])
    return SEGMENT_NAMES[seg], seg in CREDIT_SEGMENTS


new_member = {"age": 30, "visits_per_month": 4, "avg_visit_minutes": 42,
              "classes_per_month": 1, "months_member": 5, "monthly_extras_spend": 10}
new_seg, new_credit = decide(new_member)

# ---------- 5. value: member by member, all 330 members ----------
baseline_net = float(df.credit_net_if_given.sum())
df["gets_credit_v1"] = df.segment.isin(CREDIT_SEGMENTS)
v1_net = float(df.loc[df.gets_credit_v1, "credit_net_if_given"].sum())

centers = raw_centers.rename(index=relabel).sort_index()
g = df.groupby("segment")
profile = centers.round(1)
profile.insert(0, "name", [SEGMENT_NAMES[s] for s in profile.index])
profile["members"] = g.size()
profile["share_spend_under_20"] = g.monthly_extras_spend.apply(lambda s: (s < SPEND_THRESHOLD).mean()).round(2)
profile["net_if_credited_all"] = g.credit_net_if_given.sum()
profile["net_if_credited_train"] = train_seg
profile["gets_credit"] = profile.index.isin(CREDIT_SEGMENTS)
profile.to_csv(OUT / "segment_profile.csv")

# robustness: same k, clustering WITHOUT extras spend gives (almost) the same members
f2 = FEATURES[:-1]
sc2 = StandardScaler().fit(train[f2])
km2 = KMeans(n_clusters=K_CHOSEN, n_init=20, random_state=SEED).fit(sc2.transform(train[f2]))
alt = km2.predict(sc2.transform(df[f2]))
agree = pd.crosstab(df.segment, alt).max(axis=1).sum() / len(df)

k1, kc = elbow.set_index("k").loc[1], elbow.set_index("k").loc[K_CHOSEN]
results = {
    "rows_total": len(df), "rows_train": len(train), "rows_test": len(test),
    "avg_extras_spend": round(df.monthly_extras_spend.mean(), 2),
    "features": FEATURES, "left_out": ["member_id"], "scaling": "StandardScaler fit on training members",
    "k_chosen": K_CHOSEN,
    "train_spread_k1": round(k1.train_spread, 1), "train_spread_k": round(kc.train_spread, 1),
    "test_spread_k1_baseline": round(k1.test_spread, 1), "test_spread_k_v1": round(kc.test_spread, 1),
    "test_spread_reduction_pct": round(100 * (1 - kc.test_spread / k1.test_spread), 1),
    "credit_segments": [SEGMENT_NAMES[s] for s in CREDIT_SEGMENTS],
    "new_member_segment": new_seg, "new_member_gets_credit": bool(new_credit),
    "monthly_net_baseline": baseline_net, "monthly_net_v1": v1_net,
    "monthly_improvement": v1_net - baseline_net,
    "members_credited_baseline": len(df), "members_credited_v1": int(df.gets_credit_v1.sum()),
    "segment_agreement_without_spend_pct": round(100 * agree, 1),
}
(OUT / "results.json").write_text(json.dumps(results, indent=2))
df.to_csv(OUT / "members_with_segments.csv", index=False)

pd.set_option("display.width", 200)
print(elbow.round(1).to_string(index=False), "\n")
print(profile.to_string(), "\n")
print(json.dumps(results, indent=2))
