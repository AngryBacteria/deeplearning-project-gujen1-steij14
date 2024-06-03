import os
import shutil

BASE_FOLDER = "imagenet"
OUTPUT_FOLDER = "data"

# classification types
data = {}
with open(os.path.join(BASE_FOLDER, "LOC_synset_mapping.txt"), 'r') as file:
    lines = file.readlines()
    for line in lines:
        parsed = line.strip().split(" ")
        class_id = parsed[0]
        synset = "".join(parsed[1:])
        synset_splits = synset.split(",")
        synset_splits = [x.strip() for x in synset_splits]

        data[class_id] = {
            "class": synset_splits[0],
            "synset": synset,
            "synonyms": synset_splits[1:]
        }

# Bears, pandas, teddy bear
classes_to_pick = [
    "n02132136",
    "n02133161",
    "n02134084",
    "n02134418",
    "n02509815",
    "n02510455",
    "n04399382"
]

# copy image and annotation files
for class_id in classes_to_pick:
    DATA_PATH = os.path.join(BASE_FOLDER, "ILSVRC")
    # Images
    shutil.copytree(
        os.path.join(DATA_PATH, "Data", "CLS-LOC", "train", class_id),
        os.path.join(OUTPUT_FOLDER, "images", class_id)
    )
    # Annotations
    shutil.copytree(
        os.path.join(DATA_PATH, "Annotations", "CLS-LOC", "train", class_id),
        os.path.join(OUTPUT_FOLDER, "annotations", class_id)
    )

# copy mapping files
shutil.copy(
    os.path.join(BASE_FOLDER, "LOC_synset_mapping.txt"),
    os.path.join(OUTPUT_FOLDER, "LOC_synset_mapping.txt")
)
shutil.copy(
    os.path.join(BASE_FOLDER, "LOC_train_solution.csv"),
    os.path.join(OUTPUT_FOLDER, "LOC_train_solution.csv")
)

# create zip archive
shutil.make_archive(OUTPUT_FOLDER, 'zip', OUTPUT_FOLDER)