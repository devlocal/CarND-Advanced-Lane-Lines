import os

from moviepy.editor import VideoFileClip


class VideoProcessor(object):
    """Frame-based video processor"""

    def __init__(self, frame_handler):
        """
        :param frame_handler: frame handler to apply to each signle frame
        """
        self._frame_handler = frame_handler

    def process_video(self, file_name, out_file_name=None, trim_range_seconds=None):
        """Processes video with the help of frame handler"""

        if not out_file_name:
            out_file_name = "{}-out{}".format(*os.path.splitext(file_name))

        clip = VideoFileClip(file_name)
        if trim_range_seconds:
            clip = clip.subclip(*trim_range_seconds)
        out_clip = clip.fl_image(self._frame_handler)
        out_clip.write_videofile(out_file_name, audio=False)
