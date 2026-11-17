"""Build the silent, illustrative SynCT quick-start video (no patient data)."""

from pathlib import Path
from io import BytesIO
import struct

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
OUT = HERE / "SynCT_quick_start.avi"
SIZE = (1280, 720)
FPS = 6
FONT = Path(r"C:\Windows\Fonts\arial.ttf")


def font(size, bold=False):
    return ImageFont.truetype(str(FONT), size)


def text(draw, xy, value, size=28, fill="#202938", bold=False):
    draw.text(xy, value, font=font(size, bold), fill=fill)


def box(draw, xy, fill="#ffffff", outline="#d7dfe8", radius=7):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)


def scene(number, title, subtitle, rows, active, footer):
    im = Image.new("RGB", SIZE, "#f5f7f9")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1280, 90), fill="#1d3443")
    text(d, (52, 25), "SynCT", 38, "#ffffff", True)
    text(d, (235, 35), "CT-guided PET spatial normalization | Quick start", 24, "#dce9ed")
    d.rounded_rectangle((50, 124, 156, 170), radius=5, fill="#d9eee8")
    text(d, (70, 130), f"STEP {number}", 21, "#116353", True)
    text(d, (180, 115), title, 38, "#183546", True)
    text(d, (53, 186), subtitle, 23, "#526372")

    box(d, (50, 246, 1230, 605))
    for i, (label, value) in enumerate(rows):
        y = 267 + i * 63
        if i == active:
            d.rounded_rectangle((68, y - 5, 1212, y + 51), radius=5, fill="#e5f3ee")
            d.rectangle((68, y - 5, 74, y + 51), fill="#16876a")
        text(d, (90, y), label, 24, "#3c5360", True)
        text(d, (440, y), value, 23, "#223343")
    text(d, (55, 633), footer, 21, "#526372")
    text(d, (950, 661), "Illustration | No patient data", 17, "#637685")
    return im


SCENES = [
    ("Install the module", "Add this repository to 3D Slicer and restart", [
        ("Settings", "Edit > Application Settings > Modules"),
        ("Add module path", "Directory containing SynCT.py"),
        ("Open module", "PET Quantification > SynCT"),
        ("Dependencies", "SwissSkullStripper + Slicer Python packages"),
    ], 1, "Installation details are in the README."),
    ("Download model weights", "Obtain SynthMorph weights from its official site", [
        ("Download", "https://synthmorph.io"),
        ("Place in", "mri_synthmorph/synthmorph/"),
        ("Affine model", "synthmorph.affine.2.h5"),
        ("Deform model", "synthmorph.deform.3.h5"),
    ], 1, "Joint registration requires both weight files."),
    ("Prepare input images", "Use paired CT and AV45 PET from one subject", [
        ("Patient directory", "Subject001/"),
        ("CT filename", "CT.nii"),
        ("AV45 PET filename", "AV45_PET.nii"),
        ("Template image", "MNI/GAAIN template matching the masks"),
    ], 0, "Subject001 is a folder-layout example, not a patient dataset."),
    ("Configure GAAIN AV45", "Open the '5. GAAIN AV45 SUVR' section", [
        ("Cerebellum mask", "Atlas/.../voi_CerebGry_2mm.nii"),
        ("ctx label mask", "Atlas/.../voi_ctx_2mm.nii"),
        ("CT to MNI model", "joint"),
        ("Output directory", "Choose a writable results directory"),
    ], 2, "The template and masks must correspond spatially."),
    ("Smooth and run", "Smooth normalized PET before SUVR calculation", [
        ("Smooth MNI PET", "Enabled"),
        ("FWHM X/Y/Z", "6 / 6 / 6 mm (adjustable)"),
        ("Check backend", "Check GAAIN SynthMorph backend"),
        ("Start", "Run GAAIN AV45 SUVR"),
    ], 3, "PET to CT > CT to template > normalized PET > smoothing > SUVR"),
    ("Inspect outputs", "Review images and the results CSV", [
        ("Normalized PET", "AV45_PET_MNI_BSpline.nii.gz"),
        ("Smoothed PET", "AV45_PET_MNI_BSpline_smooth6mm.nii.gz"),
        ("SUVR image", "AV45_SUVR_GAAIN_cerebellumGM.nii.gz"),
        ("Results table", "AV45_GAAIN_ctx_SUVR.csv (includes CL)"),
    ], 3, "CL = 175 * SUVR_Mean - 182. Check image alignment first."),
]


def main():
    frames = []
    for i, (title, subtitle, rows, active, footer) in enumerate(SCENES, 1):
        buffer = BytesIO()
        scene(i, title, subtitle, rows, active, footer).save(buffer, format="JPEG", quality=91)
        frames.extend([buffer.getvalue()] * (5 * FPS))

    def chunk(tag, data):
        return tag + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")

    max_size = max(map(len, frames))
    avih = struct.pack(
        "<14I", 1000000 // FPS, max_size * FPS, 0, 0x10, len(frames), 0,
        1, max_size, SIZE[0], SIZE[1], 0, 0, 0, 0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh", b"vids", b"MJPG", 0, 0, 0, 0,
        1, FPS, 0, len(frames), max_size, 0xFFFFFFFF, 0,
        0, 0, SIZE[0], SIZE[1],
    )
    strf = struct.pack(
        "<IiiHHIIiiII", 40, SIZE[0], SIZE[1], 1, 24, int.from_bytes(b"MJPG", "little"),
        SIZE[0] * SIZE[1] * 3, 0, 0, 0, 0,
    )
    hdrl = chunk(b"LIST", b"hdrl" + chunk(b"avih", avih) + chunk(b"LIST", b"strl" + chunk(b"strh", strh) + chunk(b"strf", strf)))
    movi_data = bytearray(b"movi")
    index = bytearray()
    for jpeg in frames:
        offset = len(movi_data)
        movi_data += chunk(b"00dc", jpeg)
        index += struct.pack("<4sIII", b"00dc", 0x10, offset, len(jpeg))
    payload = b"AVI " + hdrl + chunk(b"LIST", movi_data) + chunk(b"idx1", index)
    OUT.write_bytes(chunk(b"RIFF", payload))
    print(OUT)


if __name__ == "__main__":
    main()
