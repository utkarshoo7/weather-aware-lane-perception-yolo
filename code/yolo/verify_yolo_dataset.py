from pathlib import Path

base = Path("C:/Projects/Flagship_Project/datasets/detection")
img_dirs = {s: base/"images"/s for s in ("train","val","test")}
lbl_dirs = {s: base/"labels"/s for s in ("train","val","test")}

def check(split):
    imgs = sorted(p.stem for p in img_dirs[split].glob("*.jpg"))
    lbls = sorted(p.stem for p in lbl_dirs[split].glob("*.txt"))
    missing_labels = [i for i in imgs if i not in lbls]
    orphan_labels = [l for l in lbls if l not in imgs]
    return len(imgs), len(lbls), len(missing_labels), len(orphan_labels)

for s in ("train","val","test"):
    imgs, lbls, miss, orph = check(s)
    print(f"{s}: images={imgs}, labels={lbls}, missing_labels={miss}, orphan_labels={orph}")
