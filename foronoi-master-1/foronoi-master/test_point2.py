#!/usr/bin/env python3
"""
Generate side-by-side frames:
 - Tree (left)
 - Voronoi (right)

Then call frames_to_video.py automatically.
GIF generation removed completely.

This version ensures a blank initial tree frame (tree_0000.png) exists so that
the very first combined frame shows an empty beachline (correct behavior).
"""
import os
import glob
import natsort
from PIL import Image
import subprocess
import sys

# Project imports
from foronoi import Voronoi, TreeObserver, Polygon, Visualizer, VoronoiObserver
from bounding_box import bounding_box
from save_png_callback import make_png_saver


# ----------------- configuration -----------------
POINTS = [
    (1,3),(4,5),(8,2),(7,6)
]

# output locations
TREE_DIR = "frames/tree"
VOR_DIR = "frames/voronoi"
COMBINED_DIR = "frames/combined"

for d in (TREE_DIR, VOR_DIR, COMBINED_DIR):
    os.makedirs(d, exist_ok=True)



# ----------------- callbacks -----------------
def tree_callback(observer, dot):
    idx = observer.n_messages
    out = os.path.join(TREE_DIR, f"tree_{idx:04d}.png")
    dot.render(filename=out, format="png", cleanup=True)
    print("[tree] saved", out)


vor_png_callback = make_png_saver(
    output_dir=VOR_DIR,
    prefix="voronoi_",
    pad=4,
    dpi=200
)



# ----------------- run Voronoi + capture -----------------
def run_and_capture(points):
    poly = Polygon(bounding_box(points))
    v = Voronoi(poly)

    # Tree observer
    tree_obs = TreeObserver(callback=tree_callback)
    v.attach_observer(tree_obs)

    # Voronoi observer
    vor_obs = VoronoiObserver(
        visualize_steps=True,
        visualize_result=True,
        callback=vor_png_callback
    )
    v.attach_observer(vor_obs)

    # Execute algorithm
    print("[runner] Creating diagram ...")
    v.create_diagram(points)
    print("[runner] Diagram creation finished.")

    # Optional visualizer (can be removed)
    try:
        Visualizer(v, canvas_offset=1)\
            .plot_sites(show_labels=True)\
            .plot_edges(show_labels=False)\
            .plot_vertices()\
            .plot_border_to_site()\
            .show()
    except Exception:
        pass



# ----------------- combine frames (side-by-side) -----------------
def combine_side_by_side():
    # read current files
    tree_files = natsort.natsorted(glob.glob(os.path.join(TREE_DIR, "tree_*.png")))
    vor_files = natsort.natsorted(glob.glob(os.path.join(VOR_DIR, "voronoi_*.png")))

    # Ensure a blank first tree frame exists (tree_0000.png).
    # If missing, create it using the size of the first voronoi frame (if any),
    # otherwise use a reasonable default size.
    expected_blank = os.path.join(TREE_DIR, "tree_0000.png")
    if not os.path.exists(expected_blank):
        # try to size blank to first voronoi frame
        if vor_files:
            try:
                with Image.open(vor_files[0]) as im:
                    w, h = im.size
            except Exception:
                w, h = 800, 600
        else:
            w, h = 800, 600
        blank = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        blank.save(expected_blank)
        print("[fix] Added blank tree_0000.png with size", (w, h))
        # refresh tree_files list
        tree_files = natsort.natsorted(glob.glob(os.path.join(TREE_DIR, "tree_*.png")))

    t = len(tree_files)
    v = len(vor_files)

    if t == 0 and v == 0:
        print("No frames found.")
        return []

    # --------------------------------
    #   FIX: match lengths by repeating last frame
    # --------------------------------
    # (After ensuring the blank start, it's safe to assume tree_files is non-empty)
    if v > t:
        last_tree = tree_files[-1]
        for _ in range(v - t):
            tree_files.append(last_tree)
        print(f"[fix] tree frames extended: {t} → {len(tree_files)}")

    elif t > v:
        # if there are more tree frames than voronoi frames, repeat last voronoi
        # (if no voronoi frames exist, create blank voronoi placeholders)
        if not vor_files:
            # create a blank voronoi file sized like the first tree image
            try:
                with Image.open(tree_files[0]) as im:
                    w, h = im.size
            except Exception:
                w, h = 800, 600
            blank_v = os.path.join(VOR_DIR, "voronoi_0000.png")
            if not os.path.exists(blank_v):
                Image.new("RGBA", (w, h), (255, 255, 255, 255)).save(blank_v)
                print("[fix] Added blank voronoi_0000.png with size", (w, h))
            vor_files = natsort.natsorted(glob.glob(os.path.join(VOR_DIR, "voronoi_*.png")))

        last_vor = vor_files[-1]
        for _ in range(t - v):
            vor_files.append(last_vor)
        print(f"[fix] voronoi frames extended: {v} → {len(vor_files)}")

    # Now both lengths equal
    n = len(tree_files)

    combined_paths = []

    for i in range(n):
        tree_img = Image.open(tree_files[i]).convert("RGBA")
        vor_img = Image.open(vor_files[i]).convert("RGBA")

        # same height
        if tree_img.size[1] != vor_img.size[1]:
            h = max(tree_img.size[1], vor_img.size[1])

            def scale(im):
                w, _ = im.size
                new_w = int(w * (h / im.size[1]))
                return im.resize((new_w, h), Image.LANCZOS)

            tree_img = scale(tree_img)
            vor_img = scale(vor_img)

        total_w = tree_img.size[0] + vor_img.size[0]
        h = tree_img.size[1]

        combined = Image.new("RGBA", (total_w, h), (255,255,255,255))
        combined.paste(tree_img, (0,0), tree_img)
        combined.paste(vor_img, (tree_img.size[0], 0), vor_img)

        out_path = os.path.join(COMBINED_DIR, f"combined_{i:04d}.png")
        combined.convert("RGB").save(out_path, quality=95)
        combined_paths.append(out_path)

        print("[combine] saved", out_path)

    return combined_paths



# ----------------- MAIN -----------------
if __name__ == "__main__":
    run_and_capture(POINTS)
    combined = combine_side_by_side()

    print("All frames combined.")

    # -----------------------------
    #  CALL frames_to_video.py
    # -----------------------------
    script_path = os.path.join(os.path.dirname(__file__), "frames_to_video.py")

    if os.path.exists(script_path):
        try:
            print("[runner] Running frames_to_video.py ...")
            subprocess.run([
                sys.executable,
                script_path,
                "--frames-dir", "frames/combined",
                "--output", "voronoi_video.mp4",
                "--fps", "1"
            ], check=True)

            print("[runner] Video created successfully.")

        except Exception as e:
            print("[runner] Failed to run frames_to_video.py:", e)

    else:
        print("[runner] frames_to_video.py not found!")
