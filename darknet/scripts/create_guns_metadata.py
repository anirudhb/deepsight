import os

### VARIABLES
IMAGES_DIR = "data/guns-object-detection/Images"
LABELS_DIR = "data/guns-object-detection/Labels"
LABELS_OUT = "data/guns_labels.txt"
LABELS_OUT_DIR = "data/guns-object-detection/Labels-Fixed"

### CODE
with open(LABELS_OUT, "w") as lf:
    for img in os.listdir(IMAGES_DIR):
        if img.startswith("."):
            continue
        img_base = img.split(".")[0]
        img_lab = img_base + ".txt"
        with open(os.path.join(LABELS_DIR, img_lab)) as lab, open(os.path.join(LABELS_OUT_DIR, img_lab), "w") as lab2:
            lab.readline()
            lab2.writelines(lab.readlines())
        lf.write(os.path.abspath(os.path.join(IMAGES_DIR, img)) + "\n")
        print("Write: {1:50} -> {0:50}".format(os.path.join(LABELS_OUT_DIR, img_lab), os.path.join(LABELS_DIR, img_lab)))