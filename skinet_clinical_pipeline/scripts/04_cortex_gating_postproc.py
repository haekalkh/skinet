"""
E7 Stage 10b - Post-processing LENGKAP ala MRCNN untuk output M2F.
Meniru 7 tahap dari skinet_e7cmp.json (mode "main"):
  1. fusion            (bb 0.1, mask 0.1)            [same-class]
  2. border_filter     (on_border 0.25, kelas 7-10)  [fallback: tepi image; colok korteks nanti]
  3. orphan_filter     (hierarki 8->[3,4] keep_if_no_child=False, 6->[9,10] keep=True)
  4. filter            (priority_table 10x10)
  5. orphan_filter     (ulang)
  6. class_fusion      ([3,4] = nsg_complet+nsg_partiel)
  7. small_filter      (min_size 50)

CATATAN class-id: config MRCNN pakai id 1-based urutan e7cmp:
  1 tubule_sain 2 tubule_atrophique 3 nsg_complet 4 nsg_partiel 5 pac
  6 vaisseau 7 veine 8 nsg 9 intima 10 media
  (border classes [7,8,9,10] = veine,nsg,intima,media)
  (orphan 8->[3,4] = nsg->[nsg_complet,nsg_partiel]; 6->[9,10]... TAPI cek: 6=vaisseau->[intima,media]? id 9,10=intima,media. YA)

Input : folder JSON Stage-9-fused M2F (atau Stage-10) per-source
Output: folder JSON per-source (format sama) siap Stage 14
Jalankan di cluster.
"""
import argparse, json, re
from pathlib import Path
import os
import numpy as np
import cv2
from pycocotools import mask as mask_util

# ---- urutan kelas config MRCNN e7cmp (1-based) ----
MRCNN_ORDER = ["tubule_sain","tubule_atrophique","nsg_complet","nsg_partiel","pac",
               "vaisseau","veine","nsg","intima","media"]
NAME2ID = {c: i+1 for i, c in enumerate(MRCNN_ORDER)}   # 1-based spt config
ID2NAME = {i+1: c for i, c in enumerate(MRCNN_ORDER)}

# priority_table dari config (10x10, index 0-based -> class_id-1)
PRIORITY_TABLE = [
 [False,True,False,False,True,False,True,False,True,True],
 [False,False,False,False,True,False,True,False,True,True],
 [True,True,False,False,True,True,True,False,False,False],
 [True,True,False,False,True,True,True,False,False,False],
 [False,False,False,False,False,False,False,False,False,False],
 [True,True,False,False,True,False,True,False,False,False],
 [True,True,False,False,False,False,False,False,False,False],
 [True,True,False,False,True,True,True,False,False,False],
 [False,False,False,False,False,False,False,False,False,False],
 [False,False,False,False,False,False,False,False,False,False],
]
# hierarki orphan (config: "8":{contains:[3,4],keep_if_no_child:False}, "6":{contains:[9,10],keep:True})
ORPHAN_HIER = {8: {"contains": [3,4], "keep_if_no_child": False},
               6: {"contains": [9,10], "keep_if_no_child": True}}
BORDER_CLASSES = [7,8,9,10]   # veine,nsg,intima,media
CLASS_FUSION = [[3,4]]        # nsg_complet + nsg_partiel

def rle_dec(rle):
    r = {"size": rle["size"], "counts": rle["counts"].encode("ascii") if isinstance(rle["counts"],str) else rle["counts"]}
    return r
def rle_enc(m):
    r = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii"); return r

def iou_mask(a, b):
    return float(np.asarray(mask_util.iou([a],[b],[0])).reshape(-1)[0])
def iomin(a, b):
    """intersection over min-area (spt mask_threshold MRCNN = part contained)."""
    inter = mask_util.area(mask_util.merge([a,b], intersect=True))
    ma, mb = mask_util.area(a), mask_util.area(b)
    mn = min(ma, mb)
    return float(inter/mn) if mn>0 else 0.0
def bb_iou(a, b):
    ba, bb_ = mask_util.toBbox(a), mask_util.toBbox(b)
    # bbox overlap iomin
    ax1,ay1,aw,ah=ba; bx1,by1,bw,bh=bb_
    ax2,ay2,bx2,by2=ax1+aw,ay1+ah,bx1+bw,by1+bh
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    mn=min(aw*ah,bw*bh); return float(inter/mn) if mn>0 else 0.0

# ---------- 1. FUSION same-class ----------
def fusion(insts, bb_th=0.1, mask_th=0.1):
    by = {}
    for it in insts: by.setdefault(it["cid"], []).append(it)
    out=[]
    for cid, group in by.items():
        used=[False]*len(group)
        for i in range(len(group)):
            if used[i]: continue
            merged=group[i]["rle"]; sc=group[i]["score"]
            for j in range(i+1,len(group)):
                if used[j]: continue
                if bb_iou(merged,group[j]["rle"])>=bb_th and iomin(merged,group[j]["rle"])>=mask_th:
                    merged=mask_util.merge([merged,group[j]["rle"]],intersect=False)
                    sc=max(sc,group[j]["score"]); used[j]=True
            used[i]=True
            out.append({"cid":cid,"rle":merged,"score":sc})
    return out

# ---------- 2. BORDER filter (fallback: tepi image / area kosong) ----------
def border_filter(insts, H, W, on_border_th=0.25, classes=BORDER_CLASSES, cortex_rle=None):
    """Buang mask yg >on_border_th di luar 'valid area'.
    valid area = cortex mask kalau ada; kalau tidak, = seluruh image KECUALI margin tepi.
    Fallback tepi-image: anggap 'border' = menyentuh batas image (proxy kasar)."""
    out=[]
    for it in insts:
        if it["cid"] not in classes:
            out.append(it); continue
        m = mask_util.decode(it["rle"])
        area = m.sum()
        if area==0: continue
        if cortex_rle is not None:
            cm = mask_util.decode(cortex_rle)
            outside = np.logical_and(m>0, cm==0).sum()
        else:
            # fallback: bagian mask di margin tepi image (2% lebar tepi)
            mb = round(0.02*min(H,W))
            border = np.zeros((H,W),np.uint8)
            border[:mb,:]=1; border[-mb:,:]=1; border[:,:mb]=1; border[:,-mb:]=1
            outside = np.logical_and(m>0, border>0).sum()
        if (outside/area) > on_border_th:
            continue   # buang
        out.append(it)
    return out

# ---------- 3/5. ORPHAN filter (parent-child containment) ----------
def orphan_filter(insts, hier=ORPHAN_HIER, bb_th=0.3, mask_th=0.3):
    keep=[True]*len(insts)
    for parent_id, spec in hier.items():
        child_ids=spec["contains"]; keep_if_no=spec["keep_if_no_child"]
        parents=[i for i,it in enumerate(insts) if it["cid"]==parent_id]
        for pi in parents:
            if not keep[pi]: continue
            has_child=False
            for ci,it in enumerate(insts):
                if it["cid"] in child_ids and keep[ci]:
                    if bb_iou(insts[pi]["rle"],it["rle"])>=bb_th and iomin(it["rle"],insts[pi]["rle"])>=mask_th:
                        has_child=True; break
            if not has_child and not keep_if_no:
                keep[pi]=False   # parent tanpa anak -> buang
    return [it for i,it in enumerate(insts) if keep[i]]

# ---------- 4. PRIORITY filter ----------
def priority_filter(insts, table=PRIORITY_TABLE, bb_th=0.3, mask_th=0.3):
    keep=[True]*len(insts)
    for i in range(len(insts)):
        if not keep[i]: continue
        for j in range(i+1,len(insts)):
            if not keep[j]: continue
            c1,c2=insts[i]["cid"]-1,insts[j]["cid"]-1
            if not (0<=c1<10 and 0<=c2<10): continue
            if bb_iou(insts[i]["rle"],insts[j]["rle"])<bb_th: continue
            if iomin(insts[i]["rle"],insts[j]["rle"])<mask_th: continue
            # table[a][b]=True -> a menghapus b
            if table[c1][c2]: keep[j]=False
            elif table[c2][c1]: keep[i]=False; break
    return [it for i,it in enumerate(insts) if keep[i]]

# ---------- 6. CLASS FUSION (merge kelas kompatibel jadi satu) ----------
def class_fusion(insts, groups=CLASS_FUSION, bb_th=0.1, mask_th=0.1):
    # gabung instance antar-kelas dlm satu grup bila overlap (pick winner by score)
    for group in groups:
        idxs=[i for i,it in enumerate(insts) if it["cid"] in group]
        used=set()
        for a in range(len(idxs)):
            i=idxs[a]
            if i in used: continue
            for b in range(a+1,len(idxs)):
                j=idxs[b]
                if j in used: continue
                if bb_iou(insts[i]["rle"],insts[j]["rle"])>=bb_th and iomin(insts[i]["rle"],insts[j]["rle"])>=mask_th:
                    # pemenang skor tertinggi (kelasnya menang)
                    if insts[j]["score"]>insts[i]["score"]:
                        insts[i]["cid"]=insts[j]["cid"]
                    insts[i]["rle"]=mask_util.merge([insts[i]["rle"],insts[j]["rle"]],intersect=False)
                    insts[i]["score"]=max(insts[i]["score"],insts[j]["score"])
                    used.add(j)
        insts=[it for k,it in enumerate(insts) if k not in used]
    return insts

# ---------- 7. SMALL filter ----------
def small_filter(insts, min_size=50):
    return [it for it in insts if mask_util.area(it["rle"])>=min_size]


# ---------- CORTEX MASK loader (gating klinis) ----------
# Set via env SKINET_CORTEX_DIR atau --cortex-dir; PNG label 1=cortex.
CORTEX_MASK_DIR = os.environ.get("SKINET_CORTEX_DIR", "")

def load_cortex_rle(source_key, H, W, cortex_dir=None):
    """Baca cortex mask PNG {src}.png (label 1=cortex), resize ke HxW, -> RLE.
    Return None kalau tak ada file (fallback ke tepi-image di border_filter)."""
    d = cortex_dir or CORTEX_MASK_DIR
    if not d:
        return None
    png = os.path.join(d, f"{source_key}.png")
    if not os.path.exists(png):
        return None
    m = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    cortex = (m == 1).astype(np.uint8)
    if cortex.shape != (H, W):
        cortex = cv2.resize(cortex, (W, H), interpolation=cv2.INTER_NEAREST)
    if cortex.sum() == 0:
        return None
    rle = mask_util.encode(np.asfortranarray(cortex))
    return rle

def process(rec, cortex_dir=None):
    H,W = rec["height"], rec["width"]
    insts=[]
    for it in rec["instances"]:
        cid = NAME2ID.get(it["class_name"])
        if cid is None: continue
        insts.append({"cid":cid, "rle":rle_dec(it["mask_rle"]), "score":it.get("score",1.0)})
    cortex_rle = load_cortex_rle(rec["source_key"], H, W, cortex_dir)
    insts = fusion(insts, 0.1, 0.1)
    insts = border_filter(insts, H, W, 0.25, BORDER_CLASSES, cortex_rle)
    insts = orphan_filter(insts, ORPHAN_HIER, 0.3, 0.3)
    insts = priority_filter(insts, PRIORITY_TABLE, 0.3, 0.3)
    insts = orphan_filter(insts, ORPHAN_HIER, 0.3, 0.3)
    insts = class_fusion(insts, CLASS_FUSION, 0.1, 0.1)
    insts = small_filter(insts, 50)
    out_inst=[]
    for it in insts:
        r=it["rle"]; r={"size":r["size"],"counts":r["counts"].decode("ascii") if isinstance(r["counts"],bytes) else r["counts"]}
        out_inst.append({"class_name":ID2NAME[it["cid"]],"mask_rle":r,"score":it["score"]})
    return {"source_key":rec["source_key"],"height":H,"width":W,"instances":out_inst}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in-dir",required=True)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--cortex-dir",default=None,help="folder cortex mask PNG (label 1=cortex); fallback env SKINET_CORTEX_DIR")
    args=ap.parse_args()
    ind,outd=Path(args.in_dir),Path(args.out_dir); outd.mkdir(parents=True,exist_ok=True)
    files=[f for f in ind.glob("*.json") if not re.match(r"^\d+_",f.name)]
    for pf in files:
        rec=json.load(open(pf))
        if "source_key" not in rec: continue
        res=process(rec, cortex_dir=args.cortex_dir)
        json.dump(res, open(outd/pf.name,"w"))
        print(f"  {rec['source_key']}: {len(rec['instances'])} -> {len(res['instances'])}")
    print(f"SELESAI: {len(files)} source -> {outd}")

if __name__=="__main__":
    main()
