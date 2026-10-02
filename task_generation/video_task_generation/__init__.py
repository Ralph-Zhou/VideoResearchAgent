"""Video DeepResearch task generation workflow.

Synthesizes tasks specific to open-web video research in which every core
reasoning step requires processing information from video frames.
"""

from video_task_generation.data_structures import VideoEntity, VideoEntityGraph

__all__ = ["VideoEntity", "VideoEntityGraph"]
