# SynCT video storyboard

Video: [SynCT_quick_start.avi](SynCT_quick_start.avi). This silent, 30-second English walkthrough uses labels from the SynCT interface. It illustrates the workflow without patient data and is not a recording of an actual patient analysis.

| Time | Scene | Action in 3D Slicer |
| --- | --- | --- |
| 00:00-00:05 | Install | Add the directory containing `SynCT.py` to `Additional module paths`, restart, and open `SynCT` |
| 00:05-00:10 | Weights | Download both `.h5` files from SynthMorph and place them in `mri_synthmorph/synthmorph/` |
| 00:10-00:15 | Inputs | Place `CT.nii` and `AV45_PET.nii` in a subject directory; select a matching template |
| 00:15-00:20 | GAAIN settings | Select the cerebellum mask, ctx mask, `joint` model, and output directory |
| 00:20-00:25 | Run | Use the default `6/6/6 mm` smoothing, check the backend, and run |
| 00:25-00:30 | Inspect | Review normalized PET, smoothed PET, SUVR, and the CSV with a CL column |

To regenerate the video on Windows:

```powershell
python -m pip install pillow
python docs/make_demo_video.py
```

For supplementary materials, link both the video and the repository README. Describe this video as an illustrated walkthrough, not as patient processing footage.
