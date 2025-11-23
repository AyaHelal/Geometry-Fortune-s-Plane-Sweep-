#!/usr/bin/env python3
"""
frames_to_video.py

Assemble PNG frames into MP4. Ensures the *actual* last frame is shown
for exactly hold_seconds (default 3.0) by appending a looped segment of
that last padded frame using ffmpeg concat (no many extra files created).

Default frames directory changed to "frames/combined".
"""
import argparse
import shutil
import subprocess
from pathlib import Path
from PIL import Image

def pad_image_to_even_and_save(src: Path, dst: Path):
    with Image.open(src) as im:
        w, h = im.size
        need_w = (w % 2) != 0
        need_h = (h % 2) != 0
        if need_w or need_h:
            new_w = w + (1 if need_w else 0)
            new_h = h + (1 if need_h else 0)
            if im.mode in ("RGBA", "LA"):
                new = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
            else:
                new = Image.new("RGB", (new_w, new_h), (255, 255, 255))
            new.paste(im, (0, 0))
            new.save(dst)
        else:
            im.save(dst)

def prepare_frames(frames_dir: Path, tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("*.png"))
    out_files = []
    for i, p in enumerate(frames):
        dst = tmp_dir / f"frame_{i:04d}.png"
        pad_image_to_even_and_save(p, dst)
        out_files.append(dst)
    return frames, out_files

def make_video_from_frames(frames_dir="frames/combined", output_name="voronoi_video.mp4",
                           fps=30, hold_seconds=3.0, cleanup=True):
    frames_dir = Path(frames_dir).resolve()
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    orig_frames, prepared = prepare_frames(frames_dir, frames_dir / "_even_frames_tmp")
    tmp_dir = frames_dir / "_even_frames_tmp"

    if not prepared:
        raise FileNotFoundError("No PNG frames found after preparation.")

    # create a single file for the last padded frame (will be used as loop input)
    last_original = orig_frames[-1]
    last_padded_file = tmp_dir / "last_frame_padded.png"
    pad_image_to_even_and_save(last_original, last_padded_file)

    output_path = frames_dir / output_name

    # Build ffmpeg command:
    # - first input: image sequence (tmp/frame_0000.png ...)
    # - second input: looped single image (-loop 1 -t hold_seconds -i last_padded)
    # - concat the two video streams
    input_pattern = str(tmp_dir / "frame_%04d.png")
    filter_complex = f"[0:v] [1:v] concat=n=2:v=1:a=0 [v]"

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", input_pattern,
        "-loop", "1",
        "-t", str(hold_seconds),
        "-i", str(last_padded_file),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    print("[ffmpeg] Running:", " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found — please install ffmpeg and ensure it is on PATH.")

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")

    if cleanup:
        try:
            shutil.rmtree(tmp_dir)
            print("[cleanup] removed temp folder")
        except Exception as e:
            print("[cleanup] warning — could not remove temp folder:", e)

    print("[done] video created at:", output_path)
    return str(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=str, default="frames/combined")
    parser.add_argument("--output", type=str, default="voronoi_video.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-seconds", type=float, default=3.0, help="Seconds to hold last frame")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not remove temp files")
    args = parser.parse_args()

    make_video_from_frames(frames_dir=args.frames_dir,
                           output_name=args.output,
                           fps=args.fps,
                           hold_seconds=args.hold_seconds,
                           cleanup=not args.no_cleanup)

if __name__ == "__main__":
    main()
