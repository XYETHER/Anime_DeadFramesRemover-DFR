# Credits and licenses

DFR's selection logic, pipeline and Colab integration were created by **xyether**.
The root MIT license covers this project's code. It does not grant rights to
third-party software, model weights or the example anime footage.

| Component | Source | Upstream license |
| --- | --- | --- |
| Depth Anything V2 code and Small weights | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | Apache-2.0 |
| Practical-RIFE code and linked models | [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | MIT |

Depth Anything V2 code is pinned to `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
Its Small checkpoint revision is `03876f8651c73a60fe4c2c48294e09fcb6838fcf`;
SHA-256 is `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`.
The setup restores Practical-RIFE's inference script from commit
`bbfd2ea90910789a860ea3e2b32a240cd577b75e` before applying the NVENC patches.
The public demo uses RIFE v4.26. The notebook also offers v4.18 and v4.22.
Upstream RIFE weights are downloaded from the linked Google Drive files;
they are not bundled or checksum-pinned by this project.

The example is a short **Jujutsu Kaisen / Gojo** clip supplied by xyether.
The underlying anime belongs to its respective rights holders. These preview
images and videos demonstrate processing and are excluded from the code license.
No ownership of the anime is claimed. Use footage you have permission to process
and distribute.

Model repositories, weights and FFmpeg are downloaded or installed by setup,
not redistributed in this repository. Their upstream notices remain applicable.
