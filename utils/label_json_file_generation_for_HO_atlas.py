import json
from nilearn import datasets

atlas_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
atlas_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')

labels_cort = atlas_cort.labels
labels_sub = atlas_sub.labels

label_map = {}

for i in range(1, 22):
    if i in (2, 13):
        continue
    label_map[str(i)] = labels_sub[i]

for i in range(1, 49):
    label_map[str(100 + i)] = labels_cort[i]

with open("atlas_labels_HO_thr25_1mm.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=4, ensure_ascii=False)

print(f"{len(label_map)} régions écrites")  # devrait afficher 67