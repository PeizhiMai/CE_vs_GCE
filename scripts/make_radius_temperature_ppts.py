#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


ROOT = Path("/Users/cosdis/Desktop/projects/CE_GCE")
DEFAULT_INPUT_DIR = ROOT / "results" / "spinless_disk_comparison_by_radius"


APPLESCRIPT = r'''
on formulaForIndex(idx)
	if idx = 1 then
		return "Energy per particle: E/N"
	else if idx = 2 then
		return "Specific heat: C/N = -beta^2 d(E/N)/d beta"
	else if idx = 3 then
		return "Compressibility: CE [V(F(N+1)-2F(N)+F(N-1))]^-1 ; GCE beta Var(N)/V"
	else
		return "Entropy per particle: CE (E-F)/(T N) ; GCE (E-Omega-mu N)/(T N)"
	end if
end formulaForIndex

on run argv
	set outPath to item 1 of argv
	set imagePaths to items 2 thru -1 of argv

	tell application "Keynote"
		activate
		set docRef to make new document
		tell docRef
			try
				delete every slide
			end try

			repeat with idx from 1 to (count imagePaths)
				set slideRef to make new slide with properties {base slide:master slide "Blank"}
				tell slideRef
					set formulaText1 to my formulaForIndex(idx)
					set textRef1 to make new text item with properties {object text:formulaText1, position:{20, 10}, width:980, height:44}
					try
						set size of object text of textRef1 to 18
					end try
					set imageFile1 to POSIX file ((item idx of imagePaths) as text)
					make new image with properties {file:imageFile1, position:{20, 58}, width:980}
				end tell
			end repeat

			export to POSIX file outPath as Microsoft PowerPoint
			close saving no
		end tell
	end tell
end run
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one PowerPoint per radius from all temperature-axis comparison figures."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing radius_* folders with *vs_temperature.png files.",
    )
    parser.add_argument(
        "--radii",
        default="4,6,8,10,12,14,16,18,20",
        help="Comma-separated radii list.",
    )
    return parser.parse_args()


def make_deck(radius_dir: Path) -> None:
    radius_dir = radius_dir.resolve()
    images = [
        radius_dir / "energy_vs_temperature.png",
        radius_dir / "specific_heat_vs_temperature.png",
        radius_dir / "compressibility_vs_temperature.png",
        radius_dir / "entropy_vs_temperature.png",
    ]
    missing = [path for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing images for {radius_dir.name}: {missing}")

    out_path = radius_dir / f"{radius_dir.name}_temperature_figures.pptx"

    with tempfile.NamedTemporaryFile("w", suffix=".applescript", delete=False) as handle:
        handle.write(APPLESCRIPT)
        script_path = Path(handle.name)

    try:
        cmd = ["osascript", str(script_path), str(out_path), *(str(path) for path in images)]
        subprocess.run(cmd, check=True)
    finally:
        script_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    radii = [item.strip() for item in args.radii.split(",") if item.strip()]

    for radius in radii:
        radius_dir = args.input_dir / f"radius_{radius}"
        make_deck(radius_dir)
        print(f"Created {radius_dir / (radius_dir.name + '_temperature_figures.pptx')}")


if __name__ == "__main__":
    main()
