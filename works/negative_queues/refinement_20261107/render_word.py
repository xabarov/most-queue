from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import shutil
import subprocess, sys

source = Path(sys.argv[1]).resolve()
base = Path(sys.argv[2]).resolve()
for d in ["preview", "render", "pages"]:
    (base / d).mkdir(parents=True, exist_ok=True)
preview = base / "preview" / source.name
with ZipFile(source) as src, ZipFile(preview, "w", ZIP_DEFLATED) as dst:
    for info in src.infolist():
        data = src.read(info.filename)
        if info.filename.endswith(".xml"):
            root = E.fromstring(data)
            for node in root.iter():
                for key, val in list(node.attrib.items()):
                    if val == "Times New Roman":
                        node.set(key, "Liberation Serif")
            data = E.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)
        dst.writestr(info, data)
soffice = shutil.which("soffice") or shutil.which("libreoffice")
if not soffice:
    raise SystemExit("Install LibreOffice and add soffice to PATH before rendering Word.")
cmd = [
    soffice,
    "-env:UserInstallation=" + (base / "lo-profile").as_uri(),
    "--headless",
    "--convert-to",
    "pdf",
    "--outdir",
    str(base / "render"),
    str(preview),
]
subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
pdf = base / "render" / (source.stem + ".pdf")
subprocess.run(["pdftotext", "-layout", str(pdf), str(base / "preview.txt")], check=True)
subprocess.run(
    ["pdftoppm", "-scale-to", "1400", "-png", str(pdf), str(base / "pages" / "page")],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
print("PDF pages:", len([p for p in (base / "preview.txt").read_text().split("\f") if p.strip()]))
