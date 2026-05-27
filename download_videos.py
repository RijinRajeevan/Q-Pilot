"""
Q-Pilot V8 — Download clean dashcam videos for perception pipeline.

These are Creative Commons / Public Domain dashcam clips from Pexels.
They have NO dashboard visible — clean front road view with traffic.

Run: python download_videos.py
"""
import os
import sys
import urllib.request
import ssl

# Pexels free video direct download URLs (no API key required for direct links)
# These are clean front-road POV dashcam clips with multiple vehicles
VIDEOS = {
    "highway": {
        "url": "https://videos.pexels.com/video-files/2053100/2053100-sd_640_360_30fps.mp4",
        "desc": "Highway driving, multiple vehicles, clean front view",
    },
    "urban": {
        "url": "https://videos.pexels.com/video-files/854669/854669-sd_640_360_25fps.mp4",
        "desc": "Urban traffic, multiple cars, city driving",
    },
    "lane_change": {
        "url": "https://videos.pexels.com/video-files/3773486/3773486-sd_640_360_30fps.mp4",
        "desc": "Multi-lane highway driving, lane changes visible",
    },
    "emergency": {
        "url": "https://videos.pexels.com/video-files/1580507/1580507-sd_640_360_30fps.mp4",
        "desc": "Highway driving with traffic, braking scenarios",
    },
    "sharp_turn": {
        "url": "https://videos.pexels.com/video-files/3945423/3945423-sd_640_360_25fps.mp4",
        "desc": "Curved road driving, turns visible",
    },
}

# Fallback: more video options if any fail
FALLBACK_URLS = [
    "https://videos.pexels.com/video-files/856237/856237-sd_640_360_30fps.mp4",
    "https://videos.pexels.com/video-files/2614736/2614736-sd_640_360_25fps.mp4",
    "https://videos.pexels.com/video-files/5765014/5765014-sd_640_360_25fps.mp4",
]

OUTPUT_DIR = os.path.join("frontend", "public", "videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map to output filenames
FILENAME_MAP = {
    "highway": "highway.mp4",
    "urban": "urban.mp4",
    "lane_change": "lane_change.mp4",
    "emergency": "emergency.mp4",
    "sharp_turn": "sharp_turn.mp4",
}


def download_video(url: str, output_path: str, desc: str) -> bool:
    """Download a video from URL to output path."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100000:
        print(f"  [OK] Already exists: {output_path} ({os.path.getsize(output_path) // 1024}KB)")
        return True

    print(f"  -> Downloading: {desc}")
    print(f"    URL: {url}")
    print(f"    -> {output_path}")

    try:
        # Create SSL context that doesn't verify (for corporate networks)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
            total = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r    Progress: {pct:.0f}% ({downloaded // 1024}KB / {total // 1024}KB)", end="")

        size_kb = os.path.getsize(output_path) // 1024
        print(f"\n  [OK] Downloaded: {size_kb}KB")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Failed: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def main():
    print("=" * 60)
    print("Q-Pilot V8 — Clean Dashcam Video Downloader")
    print("=" * 60)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print()

    success = 0
    failed = []

    for scenario, info in VIDEOS.items():
        filename = FILENAME_MAP[scenario]
        output_path = os.path.join(OUTPUT_DIR, filename)
        print(f"\n[{scenario.upper()}]")

        ok = download_video(info["url"], output_path, info["desc"])

        if not ok and FALLBACK_URLS:
            print("  Trying fallback URL...")
            fallback = FALLBACK_URLS.pop(0)
            ok = download_video(fallback, output_path, "Fallback driving footage")

        if ok:
            success += 1
        else:
            failed.append(scenario)

    print("\n" + "=" * 60)
    print(f"Downloaded: {success}/{len(VIDEOS)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print("\nFor failed downloads, the system will use existing videos as fallback.")
    print("=" * 60)


if __name__ == "__main__":
    main()
