import sys; sys.path.append(".")
import numpy as np, pandas as pd, shapely
from shapely import STRtree, linestrings, points
from pyproj import Transformer
from src.data_loader import list_trips, load_trip

TF = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)  # UTM 36N, metres
R, K = 50.0, 8   # candidate radius (m), max candidates per GPS point

def build_trip(split, tid):
    d = load_trip(tid, split)
    tr, nd, ar, rt = d["track"], d["nodes"], d["arcs"], d["route"]
    nx_, ny_ = TF.transform(nd.lon.values, nd.lat.values)
    px, py = TF.transform(tr.lon.values, tr.lat.values)
    u, v = ar.from_node.values, ar.to_node.values 
    seg = np.stack([np.c_[nx_[u], ny_[u]], np.c_[nx_[v], ny_[v]]], axis=1)   # (A,2,2)
    lines, pts = linestrings(seg), points(px, py)

    # candidates: arcs within R metres, keep K nearest
    pi, ai = STRtree(lines).query(pts, predicate="dwithin", distance=R)
    df = pd.DataFrame(dict(obs_idx=pi, arc_id=ai, dist=shapely.distance(pts[pi], lines[ai])))
    df = df.sort_values(["obs_idx", "dist"]).reset_index(drop=True)
    df["rank"] = df.groupby("obs_idx").cumcount()
    df = df[df["rank"] < K].copy()

    # STAND-IN label: nearest candidate that lies on the reference route
    df["label"] = 0
    if rt is not None:
        onr = df[df.arc_id.isin(set(rt))]
        df.loc[onr.groupby("obs_idx")["dist"].idxmin().values, "label"] = 1

    # features
    g = df.groupby("obs_idx")["dist"]
    df["dist_minus_min"] = df.dist - g.transform("min")
    df["n_cands"] = g.transform("size")
    d_arc = seg[:, 1] - seg[:, 0]
    arc_len = np.hypot(d_arc[:, 0], d_arc[:, 1])
    arc_brg = np.degrees(np.arctan2(d_arc[:, 0], d_arc[:, 1])) % 360
    dx = np.r_[np.diff(px), px[-1] - px[-2]]; dy = np.r_[np.diff(py), py[-1] - py[-2]]
    dt = np.r_[np.diff(tr.t.values), 1.0]; dt[dt <= 0] = 1.0
    step = np.hypot(dx, dy)
    gps_brg = np.degrees(np.arctan2(dx, dy)) % 360
    o, a = df.obs_idx.values, df.arc_id.values
    hd = np.abs((gps_brg[o] - arc_brg[a] + 180) % 360 - 180)    # 0..180 deg
    stationary = step[o] < 1.0
    df["heading_diff"] = np.where(stationary, 90.0, hd)
    df["stationary"] = stationary.astype(int)
    df["arc_len"] = arc_len[a]
    df["speed"] = step[o] / dt[o]
    df["dt"] = dt[o]
    df["t"] = tr.t.values[o]
    df.insert(0, "trip", tid); df.insert(0, "split", split)
    return df

if __name__ == "__main__":
    for split in ["train", "student_test"]:
        out = pd.concat([build_trip(split, t) for t in list_trips(split)], ignore_index=True)
        out.to_parquet(f"data/features_{split}.parquet", index=False)
        cov = out.groupby(["trip", "obs_idx"]).label.max()
        print(f"{split}: rows={len(out):,}  obs={len(cov):,}  "
              f"pos_rate={out.label.mean():.3f}  candidate_miss_rate={1-cov.mean():.3f}")