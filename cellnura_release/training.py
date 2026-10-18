"""Image-independent CellNura downstream training with complete inference states."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from .models import FusionClassifier, CellularGAT, make_mlp
from .inference import (read_features, feature_columns, validate_features, file_sha256,
                        build_graphs, make_batches, collate_graphs, visual_features,
                        downstream_features, load_checkpoint, predict_frame)


def split_roles(manifest_path, dataset, outer_fold=None):
    manifest = pd.read_csv(manifest_path)
    if not {"dataset", "image_id", "split"} <= set(manifest):
        raise ValueError("Manifest requires dataset,image_id,split columns")
    names = {"pannuke": "PanNuke", "consep": "CoNSeP", "crchisto": "CRCHisto"}
    dataset = names.get(str(dataset).lower(), dataset)
    subset = manifest[manifest.dataset == dataset].copy()
    if dataset == "PanNuke":
        if outer_fold not in (1, 2, 3) or "outer_fold" not in subset:
            raise ValueError("PanNuke requires --outer-fold 1, 2 or 3 and an outer_fold manifest column")
        subset = subset[subset.outer_fold.astype(str).isin([f"fold{outer_fold}", str(outer_fold)])]
    if not len(subset) or subset.image_id.duplicated().any():
        raise ValueError("No selected manifest rows or duplicated image roles")
    if not set(subset.split) <= {"train", "val", "test"}:
        raise ValueError("Unknown split role")
    roles = {role: set(subset[subset.split == role].image_id) for role in ("train", "val", "test")}
    if not all(roles.values()):
        raise ValueError("Train, validation and test image lists must all be nonempty")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if roles[a] & roles[b]:
            raise ValueError(f"{a}/{b} image overlap")
    return roles


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: file_sha256(root/name) for name in ("models.py", "training.py", "inference.py")}


def default_config(dataset, outer_fold=None):
    return {"dataset": dataset, "outer_fold": outer_fold, "variant": "full", "n_classes": 5 if dataset == "PanNuke" else 4,
            "local_dim": 96, "global_dim": 1024, "visual_dim": 1024,
            "gat_projection": 100, "gat_hidden_per_head": 128, "gat_heads": 8, "graph_dim": 256,
            "gat_dropout": 0.4, "mlp_dropout": 0.3, "mlp_hidden": [512,256,128,64],
            "graph_radius": 40.0, "max_graph_nodes": 400, "max_graphs": 64,
            "seed": 42, "fusion_epochs": 100, "gat_epochs": 60, "mlp_epochs": 500,
            "fusion_patience": 10, "gat_patience": 8, "mlp_patience": 30,
            "fusion_lr": 1e-3, "gat_lr": 5e-3, "mlp_lr": 1e-5,
            "mlp_weight_decay": 1e-4, "fusion_batch_size": 1024, "mlp_batch_size": 64}


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def dense_stage(model, inputs_train, inputs_val, labels_train, labels_val, config, stage, weighted=False):
    epochs, patience, learning_rate = config[f"{stage}_epochs"], config[f"{stage}_patience"], config[f"{stage}_lr"]
    batch_size = config[f"{stage}_batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=config["mlp_weight_decay"] if stage == "mlp" else 0.0)
    weight = None
    if weighted:
        counts = np.bincount(labels_train.cpu().numpy(), minlength=config["n_classes"]).astype(np.float64)
        values = 1.0 / np.sqrt(np.maximum(counts, 1))
        values = (values / values.sum() * len(values)).astype(np.float32)
        weight = torch.from_numpy(values).to(labels_train.device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    best, best_state, waiting, history = -np.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(labels_train), device=labels_train.device)
        # BatchNorm requires at least two nuclei in the final MLP batch.
        batches = list(order.split(batch_size))
        if stage == "mlp" and len(batches[-1]) == 1:
            if len(batches) == 1:
                raise ValueError("MLP training requires at least two labeled train nuclei")
            batches[-2] = torch.cat([batches[-2], batches[-1]])
            batches.pop()
        for indices in batches:
            optimizer.zero_grad()
            criterion(model(*[x[indices] for x in inputs_train]), labels_train[indices]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            accuracy = (model(*inputs_val).argmax(1) == labels_val).float().mean().item()
        history.append({"epoch": epoch+1, "val_accuracy": accuracy})
        if accuracy > best:
            best, best_state, waiting = accuracy, cpu_state(model), 0
        else:
            waiting += 1
        if waiting >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return best_state, {"best_val_accuracy": best, "epochs_run": len(history), "history": history}


def train_graph_stage(frame,visual,labels,roles,config,device):
    graphs = build_graphs(frame,visual,labels,config["graph_radius"])
    train_graphs = [g for g in graphs if g["img"] in roles["train"]]
    val_graphs = [g for g in graphs if g["img"] in roles["val"]]
    gat = CellularGAT(n_classes=config["n_classes"],dropout=config["gat_dropout"]).to(device)
    optimizer = torch.optim.Adam(gat.parameters(),lr=config["gat_lr"])
    criterion = nn.CrossEntropyLoss()
    best, best_state, waiting, history = -np.inf,None,0,[]
    for epoch in range(config["gat_epochs"]):
        gat.train()
        ordered = [train_graphs[i] for i in np.random.permutation(len(train_graphs))]
        for batch in make_batches(ordered,config["max_graph_nodes"],config["max_graphs"]):
            x,adjacency,y = collate_graphs(batch,device)
            keep = y >= 0
            if not keep.any():
                continue
            optimizer.zero_grad()
            criterion(gat(x,adjacency).squeeze(0)[keep],y[keep]).backward()
            optimizer.step()
        gat.eval()
        correct,total = 0,0
        with torch.no_grad():
            for batch in make_batches(val_graphs,config["max_graph_nodes"],config["max_graphs"]):
                x,adjacency,y=collate_graphs(batch,device)
                keep=y>=0
                predicted=gat(x,adjacency).squeeze(0).argmax(1)
                correct+=int((predicted[keep]==y[keep]).sum())
                total+=int(keep.sum())
        accuracy=correct/max(total,1)
        history.append({"epoch":epoch+1,"val_accuracy":accuracy})
        if accuracy>best:
            best,best_state,waiting=accuracy,cpu_state(gat),0
        else:
            waiting+=1
        if waiting>=config["gat_patience"]:
            break
    gat.load_state_dict(best_state)
    gat.eval()
    return gat,best_state,{"best_val_accuracy":best,"history":history,"epochs_run":len(history)}


def train_frame(frame, roles, config, device="cpu"):
    """Only labeled train/val nuclei supervise; every graph belongs to one image."""
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    frame = frame.reset_index(drop=True).copy()
    columns = feature_columns(frame)
    validate_features(frame, columns)
    if "label" not in frame:
        raise ValueError("Training features need label; -1 means unmatched")
    allowed = roles["train"] | roles["val"] | roles["test"]
    if not set(frame.original_image) <= allowed:
        raise ValueError("Feature image outside selected frozen manifest")
    labels = np.full(len(frame), -1, np.int64)
    supervised = frame.original_image.isin(roles["train"] | roles["val"]).to_numpy()
    labels[supervised] = frame.loc[supervised,"label"].to_numpy(dtype=np.int64)
    if ((labels < -1) | (labels >= config["n_classes"])).any():
        raise ValueError("Train/validation label outside fixed vocabulary")
    train = (labels >= 0) & frame.original_image.isin(roles["train"]).to_numpy()
    val = (labels >= 0) & frame.original_image.isin(roles["val"]).to_numpy()
    if train.sum() < 2 or val.sum() < 1:
        raise ValueError("Need at least two labeled train nuclei and one independent validation nucleus")
    tensor = lambda values: torch.from_numpy(values).to(device)
    ytrain, yval = tensor(labels[train]), tensor(labels[val])
    local=frame[columns["mv_cols"]].to_numpy(dtype=np.float32)
    global_=frame[columns["sw_cols"]].to_numpy(dtype=np.float32)
    sc_l=StandardScaler().fit(local[train]);sc_g=StandardScaler().fit(global_[train])
    fusion=FusionClassifier(config["n_classes"]).to(device)
    fusion_state,fusion_log=dense_stage(fusion,
        [tensor(sc_l.transform(local[train])),tensor(sc_g.transform(global_[train]))],
        [tensor(sc_l.transform(local[val])),tensor(sc_g.transform(global_[val]))],
        ytrain,yval,config,"fusion")
    visual=visual_features(frame,fusion,sc_l,sc_g,columns,device)
    gat,gat_state,gat_log=train_graph_stage(frame,visual,labels,roles,config,device)
    complete=downstream_features(frame,visual,gat,columns,device,config["graph_radius"],config["max_graph_nodes"],config["max_graphs"])
    scaler=StandardScaler().fit(complete[train])
    mlp=make_mlp(complete.shape[1],config["n_classes"],config["mlp_dropout"]).to(device)
    mlp_state,mlp_log=dense_stage(mlp,[tensor(scaler.transform(complete[train]))],
        [tensor(scaler.transform(complete[val]))],ytrain,yval,config,"mlp",weighted=True)
    return {"format_version":1,"fusion":fusion_state,"gat":gat_state,"mlp":mlp_state,
            "sc_l":sc_l,"sc_g":sc_g,"scaler":scaler,"columns":columns,"config":config,
            "image_roles":{k:sorted(v) for k,v in roles.items()},
            "training_log":{"fusion":fusion_log,"gat":gat_log,"mlp":mlp_log},
            "class_ids":list(range(config["n_classes"])),"n_train_nuclei":int(train.sum()),"n_val_nuclei":int(val.sum())}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",required=True)
    parser.add_argument("--split-manifest",required=True)
    parser.add_argument("--dataset",required=True,choices=["PanNuke","CoNSeP","CRCHisto"])
    parser.add_argument("--outer-fold",type=int)
    parser.add_argument("--out",required=True,help="New run directory")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--resume",action="store_true",help="Reuse only an identity-verified completed run; no manifest rewriting")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--threads",type=int,default=8)
    for stage,default in [("fusion",100),("gat",60),("mlp",500)]:
        parser.add_argument(f"--{stage}-epochs",type=int,default=default)
    args=parser.parse_args(argv)
    if min(args.fusion_epochs,args.gat_epochs,args.mlp_epochs,args.threads)<1:
        parser.error("epochs and threads must be positive")
    torch.set_num_threads(args.threads)
    roles=split_roles(args.split_manifest,args.dataset,args.outer_fold)
    config=default_config(args.dataset,args.outer_fold)
    config.update(seed=args.seed,fusion_epochs=args.fusion_epochs,gat_epochs=args.gat_epochs,mlp_epochs=args.mlp_epochs)
    identity={"feature_sha256":file_sha256(args.features),"split_sha256":file_sha256(args.split_manifest),"source_hashes":source_hashes(),"config":config}
    out=Path(args.out); checkpoint=out/"checkpoint.pth"; manifest=out/"run_manifest.json"
    if checkpoint.exists() or manifest.exists():
        if not args.resume or not checkpoint.exists() or not manifest.exists():
            raise FileExistsError("Run exists/incomplete; select a new --out or an identity-verified --resume")
        previous=json.loads(manifest.read_text())
        if previous["training_identity"]!=identity or previous["checkpoint_sha256"]!=file_sha256(checkpoint):
            raise ValueError("Resume identity/checkpoint mismatch; historical training manifest retained")
        print("Verified existing completed run; training manifest retained")
        return 0
    state=train_frame(read_features(args.features),roles,config,args.device)
    state["training_identity"]=identity
    out.mkdir(parents=True,exist_ok=True)
    torch.save(state,checkpoint)
    record={"training_identity":identity,"checkpoint_sha256":file_sha256(checkpoint),
            "image_roles":state["image_roles"],"training_log":state["training_log"],
            "n_train_nuclei":state["n_train_nuclei"],"n_val_nuclei":state["n_val_nuclei"]}
    manifest.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    loaded,fusion,gat,mlp=load_checkpoint(checkpoint,args.device)
    frame=read_features(args.features)
    predict_frame(frame[frame.original_image.isin(roles["test"])],loaded,fusion,gat,mlp,args.device).to_csv(out/"test_predictions.csv",index=False)
    print(f"Saved complete inference checkpoint and independent test predictions to {out}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
