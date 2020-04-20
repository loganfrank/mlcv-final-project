import os
from shutil import copyfile

TARGET_DIR = "images"
RESULTS_DIR = "../results/agriculture/"

if __name__ == "__main__":
    phases = ["train", "test", "val"]
    for phase in phases:
        target_phase_dir = os.path.join(TARGET_DIR, phase)
        os.makedirs(target_phase_dir, exist_ok=True)
        for root, dirs, files in os.walk(os.path.join(RESULTS_DIR, phase + "_latest", "images")):
            for fname in files:
                break_apart = fname.split("_")
                if break_apart[-2] == "synthesized":
                    copyfile(os.path.join(root, fname), os.path.join(target_phase_dir, fname))
                    print(f"{fname} copied")