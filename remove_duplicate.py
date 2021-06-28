import os
import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm


base_dir = "E:/dataset/msm1_align/imgs_remove/"
d_threshold = 5     # threshold, how similar the images are
remove = False      # delete similar images or just show it
verbose = False

for subdir in tqdm(os.listdir(base_dir), position=0, leave=True):
    path = os.path.join(base_dir, subdir)

    pairs = []
    all_pairs = []
    remove_count = 0
    i, j = 0, 0
    files = os.listdir(path)

    if len(files) == 0:
        continue

    if verbose:
        print(f"Number of files ({subdir}): {len(files)}")

    # loop to delete masked image
    for f in os.listdir(path):
        if "_" in f:
            f_path = os.path.join(path, f)
            if os.path.isfile(f_path):
                os.remove(f_path)

    files = os.listdir(path)

    if verbose:
        print(f"Number of files after remove masked image ({subdir}): {len(files)}")

    # loop to detect duplicated images
    while i < len(files):
        file_1 = os.path.join(path, files[i])
        image_1 = Image.open(file_1)
        dhash_1 = imagehash.dhash(image_1)
        j = i + 1
        pairs = [files[i]]
        while j < len(files):
            file_2 = os.path.join(path, files[j])
            image_2 = Image.open(file_2)
            dhash_2 = imagehash.dhash(image_2)      # calculate dhash
            if dhash_1 - dhash_2 <= d_threshold:
                pairs.append(files[j])
                files.remove(files[j])
                j -= 1
            j += 1
        i += 1
        if pairs:
            all_pairs.append(pairs)

    # print(all_pairs)

    # delete duplicated image
    if remove:
        for ps in all_pairs:
            while len(ps) > 1:
                file = ps.pop()
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    remove_count += 1
        print(f"Found {remove_count} duplicates")
    else:
        for ps in all_pairs:
            montage = None
            size = len(ps)
            for p in ps:
                img = cv2.imread(os.path.join(path, p))
                if montage is None:
                    montage = img
                else:
                    montage = np.hstack([montage, img])

            cv2.namedWindow('Montage', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Montage', 112*size, 112)
            cv2.imshow("Montage", montage)
            cv2.waitKey(0)