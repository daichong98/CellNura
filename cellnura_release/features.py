import argparse
import contextlib
import hashlib
import json
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from .paths import load_config, get_path
from .models import MORPH_COLS

RING_COLS = ["FR1_2"] + [f"FR{i}" for i in range(3,38)]

def morph_feats(patch_bgr):
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    props = regionprops(mask)[0]
    area, perim = props.area, props.perimeter
    maj, minor = props.major_axis_length, props.minor_axis_length
    circ = 4 * np.pi * area / perim ** 2 if perim > 0 else 0
    aspect = maj / minor if minor > 0 else 0
    convex_area = props.convex_area
    hull = cv2.convexHull(cnt)
    cper = cv2.arcLength(hull, True) if len(hull) >= 2 else 0
    it = props.inertia_tensor_eigvals
    conv = convex_area / area if area > 0 else 0
    irr = perim / cper if cper > 0 else 0
    return {"Area": float(area), "Perimeter": float(perim), "Circularity": float(circ),
            "Aspect_Ratio": float(aspect), "Convex_Area": float(convex_area),
            "Solidity": float(props.solidity), "Equivalent_Diameter": float(props.equivalent_diameter),
            "Eccentricity": float(props.eccentricity), "Bbox_Area": float(props.bbox_area),
            "Extent": float(props.extent), "Major_Axis_Length": float(maj),
            "Minor_Axis_Length": float(minor), "Orientation": float(props.orientation),
            "Inertia_Tensor_Eigvals_X": float(it[0] if len(it) > 0 else 0),
            "Inertia_Tensor_Eigvals_Y": float(it[1] if len(it) > 1 else 0),
            "Convexity": float(conv), "Irregularity_Index": float(irr)}

def ring_feats(patch_bgr):
    import math
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    dist = distance_transform_edt(mask)
    maxd = int(math.ceil(dist.max()))
    if maxd == 0:
        return None
    red = patch_bgr[:, :, 0]
    def fr(ring_mask):
        a = int(ring_mask.sum())
        if a <= 0:
            return 0.0
        px = red[ring_mask]
        thr = px.mean() + 0.5 * px.std()
        return float((px > thr).sum() / a)
    out = {}
    out["FR1_2"] = fr((dist > 0) & (dist < 2))
    ring_num, cur = 3, 2
    while cur <= maxd and ring_num <= 37:
        end = min(cur + 1, maxd + 1)
        out[f"FR{ring_num}"] = fr((dist >= cur) & (dist < end))
        ring_num += 1
        cur += 1
    while ring_num <= 37:
        out[f"FR{ring_num}"] = 0.0
        ring_num += 1
    return out


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remap_swin_key(key):
    match = re.match(r"^layers\.(\d+)\.downsample\.(.+)$", key)
    if match:
        return f"layers_{int(match.group(1))+1}.downsample.{match.group(2)}"
    return re.sub(r"^layers\.(\d+)\.", r"layers_\1.", key)


def load_swin(checkpoint, device, expected_sha):
    import timm
    import torch
    if sha256(checkpoint) != expected_sha:
        raise ValueError("Swin checkpoint SHA256 differs from the declared checkpoint")
    model = timm.create_model("swin_base_patch4_window12_384", pretrained=False,
                              features_only=True, num_classes=0)
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original = original.get("model", original)
    remapped = {remap_swin_key(k):v for k,v in original.items() if "head" not in k}
    target = model.state_dict()
    selected = {k:v for k,v in remapped.items() if k in target and v.shape == target[k].shape}
    # Fail on every missing backbone tensor, including downsample and patch_embed.
    model.load_state_dict(selected, strict=True)
    return model.to(device).eval(), {"checkpoint_sha256": expected_sha,
                                   "loaded_tensors": len(selected), "target_tensors": len(target)}


def load_mobilevit(checkpoint, device, expected_sha):
    from transformers import MobileViTForImageClassification, MobileViTImageProcessor
    names = ("config.json", "preprocessor_config.json", "model.safetensors")
    missing = [name for name in names if not (Path(checkpoint)/name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete local MobileViT checkpoint: {missing}")
    actual = {name:sha256(Path(checkpoint)/name) for name in names}
    if actual != expected_sha:
        raise ValueError("MobileViT files differ from the declared frozen checkpoint/processor identity")
    model = MobileViTForImageClassification.from_pretrained(checkpoint, output_hidden_states=True, local_files_only=True)
    processor = MobileViTImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    if getattr(model.config, "model_type", None) != "mobilevit":
        raise ValueError("Expected the declared MobileViTForImageClassification backbone")
    model = model.to(device).eval()
    if str(device).startswith("cuda"):
        model = model.half()
    return model, processor, actual


def crop_nucleus(image, nucleus):
    contour = np.asarray(nucleus.get("contour", []), np.int32)
    cx, cy = map(float, nucleus["centroid"])
    if contour.size and len(contour) >= 3:
        x,y,w,h = cv2.boundingRect(contour)
    else:
        x,y,w,h = int(cx)-10,int(cy)-10,20,20
    return image[max(0,y-10):min(image.shape[0],y+h+10),
                 max(0,x-10):min(image.shape[1],x+w+10)]


def mobilevit_crop(patch):
    # Reproduce cv2.imwrite(patch) -> PIL.open(...).convert('RGB') losslessly.
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).resize((64,64), Image.Resampling.BILINEAR)


def read_index(path, required):
    index = pd.read_csv(path, dtype={"image_id": str})
    if not set(required) <= set(index) or index.image_id.duplicated().any():
        raise ValueError(f"{path} requires unique image_id and columns {required}")
    if index[list(required)].isna().any().any():
        raise ValueError(f"Missing resource identity/path in {path}")
    return index.set_index("image_id", drop=False), Path(path).resolve().parent


def resource_path(value, index_dir):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else index_dir/path


def validate_image_references(index, index_dir, image_ids):
    """Reject the same explicit physical tile under multiple frozen image IDs."""
    seen = {}
    for image_id in sorted(set(image_ids)&set(index.index)):
        row = index.loc[image_id]
        path = resource_path(row.path,index_dir).resolve()
        position = None
        if path.suffix.lower() == ".npy":
            if "array_index" not in row or pd.isna(row.array_index):
                raise ValueError("An image array requires array_index")
            value = float(row.array_index)
            if not np.isfinite(value) or value != int(value) or value < 0:
                raise ValueError("array_index must be a finite nonnegative integer; no silent truncation")
            position = int(value)
        key = (str(path),position)
        if key in seen:
            raise ValueError(f"Frozen IDs {seen[key]} and {image_id} point to the same physical tile")
        seen[key] = image_id


def read_image(row, index_dir, array_cache):
    path = resource_path(row.path, index_dir)
    if path.suffix.lower() == ".npy":
        if "array_index" not in row or pd.isna(row.array_index):
            raise ValueError("An image array requires array_index")
        position = float(row.array_index)
        if not np.isfinite(position) or position != int(position) or position < 0:
            raise ValueError("array_index must be a finite nonnegative integer; no silent truncation")
        if path not in array_cache:
            array_cache[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        image = np.asarray(array_cache[path][int(position)])
    else:
        image = np.asarray(Image.open(path).convert("RGB"))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC three-channel image: {path}")
    return np.clip(image,0,255).astype(np.uint8)


def read_nuclei(path):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if "tiles" in data:
        if len(data["tiles"]) != 1:
            raise ValueError("Segmentation JSON must represent one frozen image, not multiple tiles")
        return data["tiles"][0]["nuc"]
    if "nuc" in data:
        return data["nuc"]
    raise ValueError(f"Missing nuc dictionary in {path}")


def global_vector(image, model, device):
    import torch
    resized = np.asarray(Image.fromarray(image).resize((384,384),Image.Resampling.LANCZOS)).astype(np.float32)/255
    normalized = (resized-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    tensor = torch.from_numpy(normalized.transpose(2,0,1).copy())[None].to(device)
    with torch.no_grad():
        last = model(tensor)[-1]
    if last.shape[-1] != 1024:
        raise ValueError(f"Expected timm final NHWC channels=1024, got {tuple(last.shape)}")
    return last.mean(dim=(1,2))[0].float().cpu().numpy()


def local_vectors(patches, model, processor, device):
    import torch
    inputs = {k:v.to(device) for k,v in processor(images=patches,return_tensors="pt").items()}
    precision = torch.autocast("cuda",dtype=torch.float16) if str(device).startswith("cuda") else contextlib.nullcontext()
    with torch.no_grad(), precision:
        states = model(**inputs,output_hidden_states=True).hidden_states[-1]
    values = states.mean(dim=(2,3)).float().cpu().numpy()
    if values.shape != (len(patches),96):
        raise ValueError(f"Expected MobileViT hidden-state spatial mean with 96 features, got {values.shape}")
    return values


def main(argv=None):
    from .training import split_roles
    parser = argparse.ArgumentParser(description="Explicit raw image/segmentation indexes to CellNura features")
    parser.add_argument("--config")
    parser.add_argument("--image-index")
    parser.add_argument("--segmentation-index")
    parser.add_argument("--split-manifest")
    parser.add_argument("--training-ground-truth",help="CSV: image_id,instance_id,cx,cy,class_id; defaults to paths.training_ground_truth; only train/val rows used")
    parser.add_argument("--dataset",required=True,choices=["PanNuke","CoNSeP","CRCHisto"])
    parser.add_argument("--outer-fold",type=int)
    parser.add_argument("--split",choices=["all","train","val","test"],default="all")
    parser.add_argument("--swin-checkpoint")
    parser.add_argument("--mobilevit-checkpoint")
    parser.add_argument("--out",required=True,help="New output CSV or trusted-local .pkl; existing files are refused")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--batch-size",type=int,default=256)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    resource = lambda explicit,key: Path(explicit) if explicit else get_path(key,config,required=True)
    output = Path(args.out)
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Use a new output path; feature identities must not be silently replaced")
    if output.suffix.lower() not in (".csv",".pkl") or args.batch_size <= 0:
        raise ValueError("Output must be .csv/.pkl and batch size must be positive")
    split_path = resource(args.split_manifest,"split_manifest")
    roles = split_roles(split_path,args.dataset,args.outer_fold)
    image_index_path = resource(args.image_index,"image_index")
    segmentation_index_path = resource(args.segmentation_index,"segmentation_index")
    images,image_dir = read_index(image_index_path,["image_id","path"])
    segments,segment_dir = read_index(segmentation_index_path,["image_id","path"])
    selected = set.union(*roles.values()) if args.split == "all" else roles[args.split]
    for index in [images,segments]:
        if selected-set(index.index):
            raise ValueError(f"Index missing frozen images: {sorted(selected-set(index.index))[:10]}")
    validate_image_references(images,image_dir,set.union(*roles.values()))
    ground_truth_path = resource(args.training_ground_truth,"training_ground_truth")
    gt = pd.read_csv(ground_truth_path,dtype={"image_id":str})
    if not {"image_id","cx","cy","class_id"} <= set(gt):
        raise ValueError("Training GT requires image_id,cx,cy,class_id")
    permitted = roles["train"]|roles["val"]
    gt = gt[gt.image_id.isin(permitted)]
    n_classes = 5 if args.dataset == "PanNuke" else 4
    if len(gt) and (gt[["cx","cy","class_id"]].isna().any().any() or not np.isfinite(gt[["cx","cy","class_id"]]).all().all()):
        raise ValueError("Nonfinite training ground truth")
    if len(gt) and ((gt.class_id != gt.class_id.astype(int)).any() or not gt.class_id.between(0,n_classes-1).all()):
        raise ValueError("Training labels must follow the declared real-class vocabulary")
    ground_truth = {image:g for image,g in gt.groupby("image_id")}
    swin_path = resource(args.swin_checkpoint,"swin_checkpoint")
    mv_path = resource(args.mobilevit_checkpoint,"mobilevit_checkpoint")
    swin,swin_info = load_swin(swin_path,args.device,config["backbones"]["swin"]["sha256"])
    mv,processor,mv_info = load_mobilevit(mv_path,args.device,config["backbones"]["mobilevit"]["file_sha256"])
    rows,ledger,array_cache = [],[],{}
    tau = {"PanNuke":12,"CoNSeP":5,"CRCHisto":10}[args.dataset]
    for number,image_id in enumerate(sorted(selected),1):
        image = read_image(images.loc[image_id],image_dir,array_cache)
        nuclei = read_nuclei(resource_path(segments.loc[image_id].path,segment_dir))
        global_ = global_vector(image,swin,args.device)
        role = next(role for role,ids in roles.items() if image_id in ids)
        pending,patches,dropped = [],[],0
        def flush():
            if not patches:
                return
            vectors = local_vectors(patches,mv,processor,args.device)
            for row,vector in zip(pending,vectors):
                row.update({f"mobilevit_feature_{i}":float(v) for i,v in enumerate(vector)})
                rows.append(row)
            patches.clear()
            pending.clear()
        for nucleus_id,nucleus in nuclei.items():
            patch = crop_nucleus(image,nucleus)
            morph = morph_feats(patch) if patch.size else None
            rings = ring_feats(patch) if patch.size else None
            if morph is None or rings is None:
                dropped += 1
                continue
            cx,cy = map(float,nucleus["centroid"])
            label = -1
            g = ground_truth.get(image_id)
            if role in ("train","val") and g is not None and len(g):
                distances = cdist([(cx,cy)],g[["cx","cy"]].to_numpy())[0]
                nearest = int(np.argmin(distances))
                if distances[nearest] <= tau:
                    label = int(g.iloc[nearest].class_id)
            row = {"original_image":image_id,"nucleus_id":int(nucleus_id),"cx":cx,"cy":cy,"role":role,"label":label}
            row.update(morph)
            row.update(rings)
            row.update({f"swin_global_{i}":float(v) for i,v in enumerate(global_)})
            pending.append(row)
            patches.append(mobilevit_crop(patch))
            if len(patches) >= args.batch_size:
                flush()
        flush()
        ledger.append({"image_id":image_id,"split":role,"segmentation_instances":len(nuclei),
                       "feature_instances":len(nuclei)-dropped,"dropped_by_feature_extraction":dropped})
        print(f"{number}/{len(selected)} {image_id}: {len(nuclei)-dropped}/{len(nuclei)}",flush=True)
    columns = ["original_image","nucleus_id","cx","cy","role","label"]+MORPH_COLS+RING_COLS+[f"swin_global_{i}" for i in range(1024)]+[f"mobilevit_feature_{i}" for i in range(96)]
    frame = pd.DataFrame(rows,columns=columns)
    if len(frame) and not np.isfinite(frame.drop(columns=["original_image","role"]).to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite extracted feature; no output written")
    output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_pickle(output) if output.suffix == ".pkl" else frame.to_csv(output,index=False)
    pd.DataFrame(ledger).to_csv(output.with_suffix(".coverage.csv"),index=False)
    info = {"dataset":args.dataset,"outer_fold":args.outer_fold,"split":args.split,"feature_sha256":sha256(output),
            "split_manifest_sha256":sha256(split_path),"swin":swin_info,"mobilevit":mv_info,
            "test_label_assignment_count":0,"image_count":len(selected),"nucleus_count":len(frame),
            "channel_policy":"historical_v35_opencv_png_roundtrip","device":args.device,
            "source_sha256":sha256(__file__),"training_ground_truth_sha256":sha256(ground_truth_path)}
    info["image_index_sha256"] = sha256(image_index_path)
    info["segmentation_index_sha256"] = sha256(segmentation_index_path)
    output.with_suffix(".manifest.json").write_text(json.dumps(info,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
