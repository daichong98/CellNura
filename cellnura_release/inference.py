import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
from .models import FusionClassifier, CellularGAT, make_mlp, MORPH_COLS

PRED_COLUMNS = ["original_image", "nucleus_id", "cx", "cy", "pred"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_features(path):
    path = Path(path)
    if path.suffix.lower() in (".pkl", ".pickle"):
        return pd.read_pickle(path)  # Trusted, locally generated data only.
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Features must be a CSV or a trusted pickle DataFrame")


def feature_columns(frame):
    if any(c.endswith("_swin") for c in frame.columns):
        raise ValueError("Conflicting cached Swin columns; rebuild features without suffix columns")
    columns = {"mv_cols": [c for c in frame if c.startswith("mobilevit_feature_")],
               "sw_cols": [c for c in frame if c.startswith("swin_global_")],
               "mo_cols": [c for c in MORPH_COLS if c in frame],
               "fr_cols": [c for c in frame if c.startswith("FR")]}
    expected = {"mv_cols": 96, "sw_cols": 1024, "mo_cols": 17, "fr_cols": 36}
    if any(len(columns[k]) != n for k, n in expected.items()):
        raise ValueError(f"Expected 96 local, 1024 global, 17 morph, 36 ring features; got { {k: len(v) for k,v in columns.items()} }")
    columns["feat_cols"] = ([f"coattention_feature_{i}" for i in range(1024)] +
                            columns["mo_cols"] + columns["fr_cols"] +
                            [f"graph_feature_{i}" for i in range(256)])
    return columns


def validate_features(frame, columns):
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate feature columns")
    required = ["original_image", "nucleus_id", "cx", "cy"]
    required += columns["mv_cols"] + columns["sw_cols"] + columns["mo_cols"] + columns["fr_cols"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if any(c.endswith("_swin") for c in frame.columns):
        raise ValueError("Conflicting cached Swin columns")
    if frame[["original_image", "nucleus_id"]].isna().any().any():
        raise ValueError("Missing image/nucleus identity")
    if frame.duplicated(["original_image", "nucleus_id"]).any():
        raise ValueError("Duplicate image/nucleus keys")
    if len(frame) and not np.isfinite(frame[[c for c in required if c != "original_image"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite coordinates/features")


def build_graphs(frame, visual, labels=None, radius=40.0):
    """Whole-image graphs; indices always refer to a reset DataFrame index."""
    graphs = []
    for image, group in frame.groupby("original_image", sort=True):
        indices = group.index.to_numpy()
        coordinates = group[["cx", "cy"]].to_numpy()
        adjacency = (cdist(coordinates, coordinates) < radius).astype(np.float32)
        np.fill_diagonal(adjacency, 1)
        graphs.append({"img": image, "indices": indices, "ids": group.nucleus_id.tolist(),
                       "x": visual[indices], "adj": adjacency,
                       "y": np.full(len(group), -1, np.int64) if labels is None else labels[indices]})
    return graphs


def make_batches(graphs, max_nodes=400, max_graphs=64):
    batches, current, nodes = [], [], 0
    for graph in graphs:
        n = len(graph["ids"])
        if current and (nodes + n > max_nodes or len(current) >= max_graphs):
            batches.append(current)
            current, nodes = [], 0
        current.append(graph)
        nodes += n
    if current:
        batches.append(current)
    return batches


def collate_graphs(graphs, device):
    sizes = [len(g["ids"]) for g in graphs]
    total = sum(sizes)
    adjacency = np.zeros((total, total), np.float32)
    offset = 0
    for graph, n in zip(graphs, sizes):
        adjacency[offset:offset+n, offset:offset+n] = graph["adj"]
        offset += n
    return (torch.from_numpy(np.concatenate([g["x"] for g in graphs])).unsqueeze(0).to(device),
            torch.from_numpy(adjacency).unsqueeze(0).to(device),
            torch.from_numpy(np.concatenate([g["y"] for g in graphs])).to(device))


def visual_features(frame, fusion, sc_l, sc_g, columns, device, batch_size=8192):
    pieces = []
    with torch.no_grad():
        for offset in range(0, len(frame), batch_size):
            group = frame.iloc[offset:offset+batch_size]
            local = sc_l.transform(group[columns["mv_cols"]].to_numpy(dtype=np.float32))
            global_ = sc_g.transform(group[columns["sw_cols"]].to_numpy(dtype=np.float32))
            pieces.append(fusion.fusion(torch.from_numpy(local).to(device), torch.from_numpy(global_).to(device)).cpu().numpy())
    return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros((0, 1024), np.float32)


def downstream_features(frame, visual, gat, columns, device, radius=40.0, max_nodes=400, max_graphs=64):
    embedding = np.empty((len(frame), 256), np.float32)
    graphs = build_graphs(frame, visual, radius=radius)
    with torch.no_grad():
        for batch in make_batches(graphs, max_nodes, max_graphs):
            x, adjacency, _ = collate_graphs(batch, device)
            values = gat.embed(x, adjacency).squeeze(0).cpu().numpy()
            offset = 0
            for graph in batch:
                count = len(graph["ids"])
                embedding[graph["indices"]] = values[offset:offset+count]
                offset += count
    shallow = frame[columns["mo_cols"] + columns["fr_cols"]].to_numpy(dtype=np.float32)
    return np.concatenate([visual, shallow, embedding], axis=1).astype(np.float32)


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=False)  # Own complete checkpoints contain sklearn states.
    required = {"fusion", "gat", "mlp", "sc_l", "sc_g", "scaler", "columns", "config", "format_version", "image_roles"}
    if not required <= set(state) or state["format_version"] != 1:
        raise ValueError("Incomplete/unsupported checkpoint; export a complete release checkpoint")
    config = state["config"]
    if config.get("variant", "full") != "full":
        raise ValueError("Only a full CellNura checkpoint is supported")
    if any(state[k] is None for k in ("fusion", "gat", "sc_l", "sc_g", "scaler")):
        raise ValueError("Incomplete checkpoint; fusion, graph, and preprocessing states are required")
    fusion = FusionClassifier(config["n_classes"]).to(device)
    gat = CellularGAT(n_classes=config["n_classes"], dropout=config["gat_dropout"]).to(device)
    mlp = make_mlp(len(state["columns"]["feat_cols"]), config["n_classes"], config["mlp_dropout"]).to(device)
    fusion.load_state_dict(state["fusion"], strict=True)
    gat.load_state_dict(state["gat"], strict=True)
    mlp.load_state_dict(state["mlp"], strict=True)
    fusion.eval()
    gat.eval()
    return state, fusion, gat, mlp.eval()


def mlp_class_ids(logits, n_classes):
    """MLP argmax over K real-class logits. The result is 0..K-1, never -1.

    Subtracting 1 would turn class 0 into unknown and drop class K-1. The frozen
    prediction files contain both -1 and every real class, so that shift does not
    match them. A detection with no MLP row is recorded as -1 by evaluation_table.
    """
    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise ValueError(f"MLP logits must have {n_classes} real-class columns, got {tuple(logits.shape)}")
    classes = logits.argmax(1)
    if classes.numel() and (int(classes.min()) < 0 or int(classes.max()) >= n_classes):
        raise ValueError("MLP argmax left the real-class range 0..K-1")
    return [int(value) for value in classes.detach().cpu().tolist()]


def read_detections(path):
    """Detection list used to place one row for every segmented nucleus."""
    frame = pd.read_csv(path)
    if "image_id" not in frame.columns and "original_image" in frame.columns:
        frame = frame.rename(columns={"original_image": "image_id"})
    required = ["image_id", "instance_id", "cx", "cy"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Detection list lacks {missing}")
    return frame[required].copy()


def evaluation_table(classified, detections):
    """Build image_id,instance_id,cx,cy,class_id rows for one detection list.

    Nuclei scored by the MLP keep that real class, 0..K-1. A detection with no
    scored row is recorded as -1. This is the only CellNura step that writes -1.
    """
    required = {"original_image", "nucleus_id", "cx", "cy", "pred"}
    missing = required - set(classified.columns)
    if missing:
        raise ValueError(f"Classified frame lacks {sorted(missing)}")
    if len(classified):
        preds = classified["pred"].to_numpy()
        if (not np.isfinite(preds).all()) or (preds < 0).any():
            raise ValueError("MLP pred must be a real class in 0..K-1; -1 is only for an unscored detection")
    rows = []
    for record in detections.itertuples(index=False):
        instance = str(record.instance_id)
        if not (instance.startswith("d") and instance[1:].isdigit()):
            raise ValueError(f"Detection instance_id must look like d0, got {instance}")
        same_image = classified["original_image"].astype(str) == str(record.image_id)
        same_point = np.isclose(classified["cx"].to_numpy(dtype=float), float(record.cx)) & np.isclose(classified["cy"].to_numpy(dtype=float), float(record.cy))
        hit = classified.loc[same_image.to_numpy() & same_point]
        if len(hit) > 1:
            raise ValueError(f"Multiple MLP rows for {record.image_id} {instance}")
        pred = -1 if hit.empty else int(hit.iloc[0]["pred"])
        rows.append({"image_id": record.image_id, "instance_id": instance,
                     "nucleus_id": int(instance[1:]), "cx": float(record.cx), "cy": float(record.cy),
                     "pred": pred, "class_id": pred})
    return pd.DataFrame(rows, columns=["image_id", "instance_id", "nucleus_id", "cx", "cy", "pred", "class_id"])


def predict_frame(frame, state, fusion, gat, mlp, device="cpu"):
    """GT/label columns are deliberately ignored; no scaler fitting occurs."""
    frame = frame.reset_index(drop=True).copy()
    columns, config = state["columns"], state["config"]
    validate_features(frame, columns)
    if not len(frame):
        return pd.DataFrame(columns=PRED_COLUMNS)
    visual = visual_features(frame, fusion, state["sc_l"], state["sc_g"], columns, device)
    complete = downstream_features(frame, visual, gat, columns, device, config["graph_radius"], config["max_graph_nodes"], config["max_graphs"])
    predictions = []
    with torch.no_grad():
        for start in range(0, len(complete), 8192):
            x = torch.from_numpy(state["scaler"].transform(complete[start:start+8192])).to(device)
            predictions.extend(mlp_class_ids(mlp(x), config["n_classes"]))
    result = frame[PRED_COLUMNS[:-1]].copy()
    result["pred"] = predictions
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--detections", help="CSV of every segmented nucleus: image_id,instance_id,cx,cy. A nucleus with no MLP row is written as class_id -1; scored nuclei stay 0..K-1.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-manifest")
    parser.add_argument("--dataset")
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args(argv)
    frame = read_features(args.features)
    state, fusion, gat, mlp = load_checkpoint(args.checkpoint, args.device)
    roles = None
    if args.split_manifest or args.split != "all":
        declared = state["config"]
        if args.dataset is not None and args.dataset != declared["dataset"]:
            raise ValueError("Frozen split dataset differs from trained checkpoint")
        if args.outer_fold is not None and args.outer_fold != declared.get("outer_fold"):
            raise ValueError("Frozen split outer fold differs from trained checkpoint")
    if args.split_manifest:
        from .training import split_roles
        roles = split_roles(args.split_manifest,state["config"]["dataset"],state["config"].get("outer_fold"))
        if any(roles[key] != set(state["image_roles"][key]) for key in ("train","val","test")):
            raise ValueError("Frozen manifest roles differ from checkpoint; trained images cannot be relabeled as test")
    if args.split != "all":
        if roles is not None:
            keep = roles[args.split]
        else:
            keep = set(state["image_roles"][args.split])
        frame = frame[frame.original_image.isin(keep)]
    result = predict_frame(frame, state, fusion, gat, mlp, args.device)
    if args.detections:
        result = evaluation_table(result, read_detections(args.detections))
        result = result[["image_id", "instance_id", "cx", "cy", "class_id"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Wrote {len(result)} predictions from complete saved states to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
