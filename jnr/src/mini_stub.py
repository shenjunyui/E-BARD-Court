import os
import shutil

src_root = "../data/stub"
dst_root = "../data/mini_stub"
max_images = 16
init = "g"
for game in os.listdir(src_root):
    game_path = os.path.join(src_root, game)
    if not os.path.isdir(game_path):
        continue

    if game[0]==init:
        continue
    print(game_path)
    for track in os.listdir(game_path):
        track_path = os.path.join(game_path, track)
        if not os.path.isdir(track_path) or not track.startswith("track_"):
            continue

        # Destination path
        dst_path = os.path.join(dst_root, game, track)
        os.makedirs(dst_path, exist_ok=True)

        # List and sort images
        images = sorted([f for f in os.listdir(track_path) if f.endswith(".png")])
        for img in images[:max_images]:
            src_img = os.path.join(track_path, img)
            dst_img = os.path.join(dst_path, img)
            shutil.copy2(src_img, dst_img)
