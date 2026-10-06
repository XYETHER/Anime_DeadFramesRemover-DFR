# 🎞️ Anime_DeadFramesRemover(DFR)

### Get your anime clips ready for smoother edits ✨

**Created by [xyether](https://github.com/XYETHER).**

Anime often holds the same drawing for several frames. **DFR** helps trim those
holds before you add smoother motion with **RIFE** — a tool that creates the
frames in between.

Upload a clip, pick a mode, and download the result. You can run it in your
browser with Google Colab — no coding needed.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XYETHER/Anime_DeadFramesRemover-DFR/blob/main/DFR_Colab.ipynb)

[📥 Download the notebook & examples](https://github.com/XYETHER/Anime_DeadFramesRemover-DFR/releases/latest)

## 🎬 Watch the difference

Here's Inumaki going through the pipeline. **Click any preview to watch its MP4.**

| 📹 Original clip | 🌈 Depth map |
| --- | --- |
| [![Input clip](assets/input.gif?v=inumaki2)](https://github.com/XYETHER/Anime_DeadFramesRemover-DFR/raw/refs/heads/main/assets/input.mp4?v=inumaki2) | [![Depth map results](assets/depth.gif?v=inumaki2)](https://github.com/XYETHER/Anime_DeadFramesRemover-DFR/raw/refs/heads/main/assets/depth.mp4?v=inumaki2) |
| The starting clip. | Helps DFR decide where to look for movement. |

| ✨ Original + RIFE | 🚀 DFR + RIFE |
| --- | --- |
| [![Input plus interpolation](assets/input-interpolated.gif?v=inumaki2)](https://github.com/XYETHER/Anime_DeadFramesRemover-DFR/raw/refs/heads/main/assets/input-interpolated.mp4?v=inumaki2) | [![DFR plus interpolation](assets/dfr-interpolated.gif?v=inumaki2)](https://github.com/XYETHER/Anime_DeadFramesRemover-DFR/raw/refs/heads/main/assets/dfr-interpolated.mp4?v=inumaki2) |
| Smoothing the original clip. | Trimming frames first, then smoothing. |

The RIFE examples use **8× interpolation at 60 FPS** for slowed playback.
DFR shortens the clip, so the two results have different lengths.
[Watch the DFR result before RIFE](assets/dfr.mp4).

## 🚀 Try it yourself

1. **Open Colab** using the button above.
2. **Choose a T4 GPU:** go to **Runtime → Change runtime type → T4 GPU**.
3. **Run “1. DFR — Setup.”** Click the ▶ button and let it finish.
4. **Add your clip** in **“2. DFR — Remove dead frames.”** Leave `Video_source` blank to upload, or paste a public Google Drive file link.
5. **Pick a mode and run it.** Start with **Type 1** and **H.264 (NVIDIA)** for easy playback, then download your `_DFR.mp4`.

💡 DFR exports a **silent video at 23.976 FPS**, keeping your clip's resolution
and colors. Its length changes — adjust timing and add audio in your editor afterward.

### 🎛️ Which mode should I choose?

| Mode | When to use it |
| --- | --- |
| **Type 1 — start here** | More aggressive frame removal. Checks every third frame. |
| **Type 2** | Keeps more frames to check. Try it if Type 1 drops movement you wanted. Checks every second frame. |

Give the result a quick watch, especially around blinks, fast action and scene changes.

### ✨ Want smoother slow motion too?

After DFR, run the bonus **RIFE SETUP** cell. In **PROCESS VIDEO**, enter the
DFR output's `/content` path and run it.

To match the examples, use:

| Setting | Value |
| --- | --- |
| `rife_model` | `v4.26` |
| `interpolation_factor` | `8` |
| `slow_down` | `60 fps` |
| `deadframes` | **Off** — DFR already handled this step |
| `encoder` | `h264_nvenc` |

You can also run RIFE on your original clip to compare the results.

## 🧠 What's happening behind the scenes?

**Your clip → DFR frame selection → optional RIFE smoothing → your edit**

DFR estimates which parts of the picture are closer to the camera, then uses
those regions to focus its movement checks. It compares the original pixels
and keeps the frames it selects.

![Original pixels, estimated depth and attention regions](assets/depth-attention.jpg?v=inumaki2)

**Left:** your clip. **Middle:** the depth estimate. **Right:** where DFR focuses
its checks. The dark background is just for this explanation — your output keeps
the full picture.

## 🔍 Curious about the details?

<details>
<summary><b>See the example settings and results</b></summary>

These previews use `inumaki2.mp4` and were generated on a **Tesla T4**.
Video processing used the original **1920 × 1080** clip; the published MP4s are
smaller **960 × 540** copies for easy viewing. GIFs show up to four seconds at 12 FPS.

| Video | Frames | FPS | Length |
| --- | ---: | ---: | ---: |
| Original | 19 | 23.976 | 0.79 s |
| Depth map | 19 | 23.976 | 0.79 s |
| Original + RIFE | 145 | 60 | 2.42 s |
| DFR + RIFE | 49 | 60 | 0.82 s |

Type 1 checked seven of the 19 source frames and kept all seven. The other 12
were skipped by the mode's frame stepping; none of the checked frames was
classified as held. This short clip is a demo, not an accuracy benchmark.

For the full breakdown, see the [frame decisions](assets/decisions.csv) and
[processing report](assets/report.json).

</details>

<details>
<summary><b>Run locally or reproduce the previews</b></summary>

Colab is the tested route for the full workflow. Local use needs an NVIDIA CUDA
GPU, CUDA-enabled PyTorch, and FFmpeg/ffprobe with NVIDIA hardware encoding.
Use SDR video; HDR is not supported.

```bash
git clone https://github.com/XYETHER/Anime_DeadFramesRemover-DFR.git
cd Anime_DeadFramesRemover-DFR
python -m pip install -r requirements.txt
python -m dfr input.mp4 --type 1 --codec h264_nvenc --output-dir results
```

Depth weights download on the first run and are cached in `.dfr-cache`.
The notebook includes the optional RIFE steps.

To reproduce the four MP4 previews inside Colab after cloning:

```bash
python scripts/generate_previews.py /content/your_clip.mp4
```

Results go to `/content/DFR_demo/previews`. The script keeps full-resolution
renders, uses Type 1, and applies RIFE v4.26 8× at 60 FPS to both video paths.

</details>

## ❤️ Credits & license

**DFR was created by [xyether](https://github.com/XYETHER).**

Thanks to [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2)
for depth estimation and [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
for interpolation.

Project code is [MIT licensed](LICENSE). Models, third-party tools and example
footage have their own terms — see [credits & notices](THIRD_PARTY_NOTICES.md).
