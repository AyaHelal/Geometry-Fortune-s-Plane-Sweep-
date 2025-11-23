# save_png_callback.py
import os
import matplotlib.pyplot as plt

def make_png_saver(output_dir="frames", prefix="frame_", pad=4, dpi=200, close_after_save=True):
    """
    Returns a callback function for VoronoiObserver that saves the provided matplotlib Figure as PNG.
    - output_dir: directory to store PNG frames (default "frames")
    - prefix: file name prefix, e.g. "frame_"
    - pad: zero-padding digits for index
    - dpi: resolution of saved PNG (increase for higher quality)
    - close_after_save: whether to close the figure after saving
    """
    os.makedirs(output_dir, exist_ok=True)

    def callback(observer, fig):
        """
        observer: the VoronoiObserver instance (we use observer.n_messages as the frame index)
        fig: matplotlib Figure (returned by Visualizer.get_canvas())
        """
        idx = observer.n_messages
        filename = f"{prefix}{idx:0{pad}d}.png"
        path = os.path.join(output_dir, filename)
        try:
            # Save as PNG directly; format inferred by extension but pass dpi explicitly.
            fig.savefig(path, format="png", bbox_inches="tight", dpi=dpi)
            print(f"[save_png] {path}")
        except Exception as e:
            print(f"[save_png] Warning: failed to save {path}: {e}")
        finally:
            if close_after_save:
                try:
                    plt.close(fig)
                except Exception:
                    pass

    return callback