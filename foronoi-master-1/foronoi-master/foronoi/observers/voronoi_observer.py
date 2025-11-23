# voronoi_observer.py (modified)
import os
from abc import ABC

from foronoi.algorithm import Algorithm
from foronoi.observers.message import Message
from foronoi.observers.observer import Observer
import matplotlib.pyplot as plt

from foronoi.visualization.visualizer import Visualizer, Presets


class VoronoiObserver(Observer, ABC):
    def __init__(self, visualize_steps=True, visualize_before_clipping=False, visualize_result=True, callback=None,
                 figsize=(8, 8), canvas_offset=1, settings=None,
                 # new options for saving frames
                 save_frames=False, frames_dir="frames", frames_prefix="frame_", close_fig_after_save=True):
        """
        Observes the state of the algorithm (:class:`foronoi.algorithm.Algorithm`) and visualizes
        the result using the Visualizer (:class:`foronoi.visualization.visualizer.Visualizer`).

        Parameters
        ----------
        visualize_steps: bool
            Visualize all individual steps
        visualize_before_clipping: bool
            Visualize the result before the edges are clipped
        visualize_result: bool
            Visualize the final result
        callback: function
            By default, the VoronoiObserver shows or prints the result when
            `text_based` is true. When a callback function is given, either the GraphViz diagram or the text-string
            is passed to the callback.

            The callback signature must be: callback(observer, fig)
            where `observer` is this VoronoiObserver instance and `fig` is a matplotlib Figure object.

        figsize: (float, float)
            Window size in inches
        canvas_offset: float
            The space around the bounding object
        settings: dict
            Visualizer settings to override the default presets used by the VoronoiObserver

        save_frames: bool
            If True, a default callback will be used to save each figure as a PNG into `frames_dir`.
            If a custom `callback` is provided, that will be used instead of the default saver.
        frames_dir: str
            Directory where frames will be saved when save_frames=True
        frames_prefix: str
            Prefix for frame filenames (index appended with zero padding)
        close_fig_after_save: bool
            Whether to close the figure after saving (helps free memory)
        """
        self.canvas_offset = canvas_offset
        self.figsize = figsize
        self.visualize_steps = visualize_steps
        self.visualize_before_clipping = visualize_before_clipping
        self.visualize_result = visualize_result
        self.n_messages = 0
        self.messages = []
        self.settings = settings or {}

        # Handle frame-saving options and default callback logic
        self._save_frames = bool(save_frames)
        self._frames_dir = frames_dir
        self._frames_prefix = frames_prefix
        self._close_fig_after_save = bool(close_fig_after_save)

        # If user provided a callback, use it.
        # Otherwise, select default: either plt.show or a frame-saver depending on save_frames.
        if callback is not None:
            self.callback = callback
        else:
            if self._save_frames:
                # ensure directory exists
                os.makedirs(self._frames_dir, exist_ok=True)

                def _saver(observer, fig):
                    """
                    Save the provided matplotlib Figure `fig` to disk using the observer's current message index.
                    Uses observer.n_messages as the index (this mirrors the previous implementation where n_messages
                    is incremented after callback is invoked).
                    """
                    idx = observer.n_messages
                    filename = f"{self._frames_prefix}{idx:04d}.png"
                    path = os.path.join(self._frames_dir, filename)
                    try:
                        fig.savefig(path, bbox_inches="tight")
                    except Exception:
                        # best-effort save; don't raise to avoid interrupting the algorithm flow
                        pass
                    finally:
                        if self._close_fig_after_save:
                            try:
                                plt.close(fig)
                            except Exception:
                                pass

                self.callback = _saver
            else:
                # default behavior: show the figure (blocking)
                self.callback = lambda a, b: plt.show(block=True)

    def update(self, subject: Algorithm, message: Message, **kwargs):
        """
        Send the updated state of the algorithm to the VoronoiObserver.

        Parameters
        ----------
        subject: Algorithm
            The algorithm to observe
        message: Message
            The message type
        """
        if not isinstance(subject, Algorithm):
            return False

        if message == Message.STEP_FINISHED and self.visualize_steps:
            vis = Visualizer(subject, canvas_offset=self.canvas_offset)
            settings = Presets.construction
            settings.update(self.settings)
            # note: original code asserted this; keep assertion to detect unexpected states
            assert subject.sweep_line == subject.event.yd
            result = vis.plot_all(**settings)
            plt.title(str(subject.event) + "\n")
        elif message == Message.SWEEP_FINISHED and self.visualize_before_clipping:
            vis = Visualizer(subject, canvas_offset=self.canvas_offset)
            settings = Presets.clipping
            settings.update(self.settings)
            result = vis.plot_all(**settings)
            plt.title("Sweep finished\n")
        elif message == Message.VORONOI_FINISHED and self.visualize_result:
            vis = Visualizer(subject, canvas_offset=self.canvas_offset)
            settings = Presets.final
            settings.update(self.settings)
            result = vis.plot_all(**settings)
            plt.title("Voronoi completed\n")
        else:
            return

        # Pass the figure (matplotlib Figure) to the callback. Callback is responsible for saving/closing if desired.
        try:
            self.callback(self, result.get_canvas())
        except TypeError:
            # In case a callback was provided that expects different args, try a fallback:
            try:
                # some callbacks might accept only the figure
                self.callback(result.get_canvas())
            except Exception:
                # swallow exceptions to avoid breaking the observed algorithm; it's better to continue.
                pass

        # Update bookkeeping (n_messages increments after the callback in the original implementation)
        self.n_messages += 1
        self.messages.append(message)