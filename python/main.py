import os

from basic_logging import setup_basic_logging
from pipeline_convolution import PipelineConvolution
from pipeline_hls import PipelineHls
from tracker import LaneTracker
from video import VideoProcessor


INPUT_VIDEO_FILES = [
    "project_video.mp4",
    "challenge_video.mp4",
    "harder_challenge_video.mp4",
]


def process_video(input_file_name, output_file_name, pipeline_name,
                  draw_contour=False, draw_windows=False, draw_frame_number=False, visualize_perspective=True,
                  trim_range_seconds=None):
    # Build single frame processing pipeline
    if pipeline_name == 'hls':
        pipeline = PipelineHls()
    elif pipeline_name == 'convolution':
        pipeline = PipelineConvolution()
    else:
        raise ValueError("Unknown pipeline name {}".format(pipeline_name))

    # Create lane tracker
    tracker = LaneTracker(
        pipeline=pipeline,
        draw_contour=draw_contour,
        draw_windows=draw_windows,
        draw_frame_number=draw_frame_number,
        visualize_perspective=visualize_perspective
    )

    # Process video file
    processor = VideoProcessor(tracker.process_frame)
    processor.process_video(
        file_name=input_file_name,
        out_file_name=output_file_name,
        trim_range_seconds=trim_range_seconds
    )


def main():
    setup_basic_logging()

    debug_kwargs = {
        "draw_contour": True,
        "draw_windows": True,
        "draw_frame_number": True,
        "visualize_perspective": False,
        "trim_range_seconds": (28, 30),
    }

    process_video(
        input_file_name=os.path.join("..", INPUT_VIDEO_FILES[1]),
        output_file_name="../temp/debug.mp4",
        pipeline_name='convolution',
        # **debug_kwargs
    )


if __name__ == "__main__":
    main()
