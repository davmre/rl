"""Utilities to save and render videos, e.g. in Jupyter notebooks.

Note that these require some additional packages not specified in the core
`requirements.txt`.
"""
from base64 import b64encode
import os

import cv2
import ffmpeg
from IPython import display
from matplotlib import animation
from matplotlib import pylab as plt  # type: ignore
import pyvirtualdisplay

_DISPLAY = None


def initialize_virtual_display(size=(1920, 1080)):
    """Initializes a virtual display for use in Jupyter notebooks.

    This enables calls to `env.render(mode="rgb_array")` for OpenAI gym
    environments in headless contexts (e.g., a Jupyter notebook server).
    """
    global _DISPLAY
    if _DISPLAY is None:
        _DISPLAY = pyvirtualdisplay.Display(  # type: ignore
            visible=False,  # use False with Xvfb
            size=size)
        _ = _DISPLAY.start()
        print("Initialized virtual display.")
    else:
        print("Display already initialized; doing nothing.")


def _write_avi(filename, frames, frames_per_second):
    height, width = frames[0].shape[:-1]
    out = cv2.VideoWriter(filename,
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          frames_per_second,
                          frameSize=(width, height))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()


def write_mp4(filename,
              frames,
              frames_per_second=15,
              temp_filename='/tmp/avifile.avi'):
    _write_avi(temp_filename, frames, frames_per_second=frames_per_second)
    if os.path.exists(filename):
        print("Removing existing file:", filename)
        os.remove(filename)
    ffmpeg.input(temp_filename).output(filename).run()
    os.remove(temp_filename)


def display_remote_video(video_path, video_width=None):
    """Renders a video stored on the server in a Jupyter notebook."""
    with open(video_path, "r+b") as f:
        video_data = f.read()

    video_url = f"data:video/mp4;base64,{b64encode(video_data).decode()}"
    return display.HTML(
        f"""<video {f"width={video_width}" if video_width else ""} controls><source src="{video_url}"></video>"""
    )


def display_frames(frames, frames_per_second=15, as_video=False):
    """Displays a sequence of frames as a Jupyter notebook animation.

    This seems to be slower than rendering to and then just displaying an
    MP4 video, but unlike that approach it works in the VSCode notebook view
    (which doesn't support videos).
    """
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])
    plt.axis('off')

    def init():
        img.set_data(frames[0])
        return (img,)

    def animate(i):
        img.set_data(frames[i])
        return (img,)

    ani = animation.FuncAnimation(
        fig,
        func=animate,  # type: ignore
        init_func=init,
        frames=len(frames),
        interval=1000 / frames_per_second,
        blit=True)
    if as_video:
        html = ani.to_html5_video()
    else:
        html = ani.to_jshtml()
    plt.close()
    return display.HTML(html)