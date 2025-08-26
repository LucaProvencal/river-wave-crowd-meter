import subprocess
import cv2
import numpy as np
import tempfile
import os

class Stream:
    def get_youtube_stream_url(self, youtube_url, yt_dlp_path="yt-dlp"):
        """
        Uses yt-dlp to fetch the direct stream URL from a YouTube live link.
        yt_dlp_path: path to the yt-dlp binary (or just 'yt-dlp' if in PATH)
        """
        try:
            result = subprocess.run(
                [yt_dlp_path, "-g", youtube_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            # Return the first URL (video-only or best available)
            stream_url = result.stdout.strip().splitlines()[0]
            return stream_url
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"yt-dlp failed: {e.stderr}")

    def grab_frame_from_stream(self, youtube_url, ffmpeg_path="/opt/homebrew/bin/ffmpeg"):
        """
        Uses FFmpeg to capture a single frame from a livestream URL.
        Returns a NumPy array (BGR) suitable for OpenCV.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            tmp_filename = tmpfile.name

        stream_url = self.get_youtube_stream_url(youtube_url)

        # FFmpeg command: grab 1 frame
        cmd = [
            ffmpeg_path,
            "-y",  # overwrite if exists
            "-i", stream_url,
            "-frames:v", "1",
            "-q:v", "2",  # quality for jpg
            tmp_filename
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            os.unlink(tmp_filename)
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        # Load image into OpenCV
        image = cv2.imread(tmp_filename)
        os.unlink(tmp_filename)
        if image is None:
            raise RuntimeError("Failed to load frame into OpenCV")
        
        return image