#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


ROOT = Path("/Users/cosdis/Desktop/projects/CE_GCE")
DEFAULT_INPUT_DIR = ROOT / "results" / "spinless_disk_comparison_by_radius"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "summary.pptx"


APPLESCRIPT = r'''
on splitText(theText, theDelimiter)
	set oldTID to AppleScript's text item delimiters
	set AppleScript's text item delimiters to theDelimiter
	set theItems to text items of theText
	set AppleScript's text item delimiters to oldTID
	return theItems
end splitText

on formulaForIndex(idx)
	if idx = 1 then
		return "Energy: E/N"
	else if idx = 2 then
		return "Specific heat: C/N = -beta^2 d(E/N)/d beta"
	else if idx = 3 then
		return "Compressibility: kappa_CE=[V(F(N+1)-2F(N)+F(N-1))]^-1 ; kappa_GCE=beta Var(N)/V"
	else
		return "Entropy: S/N ; CE=(E-F)/(TN), GCE=(E-Omega-mu N)/(TN)"
	end if
end formulaForIndex

on run argv
	set outPath to item 1 of argv
	set slideSpecs to items 2 thru -1 of argv

	tell application "Keynote"
		activate
		set docRef to make new document
		tell docRef
			try
				delete every slide
			end try

			repeat with slideSpec in slideSpecs
				set slideParts to my splitText((contents of slideSpec) as text, "|||")
				set slideTitle to item 1 of slideParts
				set img1Path to item 2 of slideParts
				set img2Path to item 3 of slideParts
				set img3Path to item 4 of slideParts
				set img4Path to item 5 of slideParts

				set slideRef to make new slide with properties {base slide:master slide "Blank"}
				tell slideRef
					set titleRef to make new text item with properties {object text:slideTitle, position:{8, 2}, width:1008, height:20}
					try
						set size of object text of titleRef to 18
					end try

					set header1 to make new text item with properties {object text:my formulaForIndex(1), position:{8, 22}, width:500, height:22}
					set header2 to make new text item with properties {object text:my formulaForIndex(2), position:{516, 22}, width:500, height:22}
					set header3 to make new text item with properties {object text:my formulaForIndex(3), position:{8, 394}, width:500, height:28}
					set header4 to make new text item with properties {object text:my formulaForIndex(4), position:{516, 394}, width:500, height:28}
					repeat with hdr in {header1, header2, header3, header4}
						try
							set size of object text of hdr to 11
						end try
					end repeat

					make new image with properties {file:(POSIX file img1Path), position:{8, 44}, width:500}
					make new image with properties {file:(POSIX file img2Path), position:{516, 44}, width:500}
					make new image with properties {file:(POSIX file img3Path), position:{8, 422}, width:500}
					make new image with properties {file:(POSIX file img4Path), position:{516, 422}, width:500}
				end tell
			end repeat

			export to POSIX file outPath as Microsoft PowerPoint
			close saving no
		end tell
	end tell
end run
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary.pptx from selected radius folders.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing radius_* folders.",
    )
    parser.add_argument(
        "--radii",
        default="4,8,12,16,20",
        help="Comma-separated radii to use for slides 1..N.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PowerPoint path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radii = [item.strip() for item in args.radii.split(",") if item.strip()]

    slide_specs: list[str] = []
    for radius in radii:
        radius_dir = (args.input_dir / f"radius_{radius}").resolve()
        images = [
            radius_dir / "energy_vs_temperature.png",
            radius_dir / "specific_heat_vs_temperature.png",
            radius_dir / "compressibility_vs_temperature.png",
            radius_dir / "entropy_vs_temperature.png",
        ]
        missing = [path for path in images if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing images for radius {radius}: {missing}")
        slide_specs.append(
            "|||".join(
                [
                    f"R = {radius}",
                    *(str(path) for path in images),
                ]
            )
        )

    with tempfile.NamedTemporaryFile("w", suffix=".applescript", delete=False) as handle:
        handle.write(APPLESCRIPT)
        script_path = Path(handle.name)

    try:
        cmd = ["osascript", str(script_path), str(args.output.resolve()), *slide_specs]
        subprocess.run(cmd, check=True)
    finally:
        script_path.unlink(missing_ok=True)

    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
