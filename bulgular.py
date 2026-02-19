from PIL import Image, ImageFont, ImageDraw
import difflib
import cv2
import os

imgname = "img10.jpeg"  # veya .jpg
base = "pilot_dataset"
orig = os.path.join(base, "images", imgname)
clahe = os.path.join(base, "outputs", "clahe_sauvola", os.path.splitext(imgname)[0]+".png")
raw_txt = open(os.path.join(base,"outputs","raw_ocr", os.path.splitext(imgname)[0]+".txt"), encoding="utf-8").read()
clahe_txt = open(os.path.join(base,"outputs","clahe_sauvola", os.path.splitext(imgname)[0]+".txt"), encoding="utf-8").read()

# Yan yana görsel
A = Image.open(orig).convert("RGB").resize((600,400))
B = Image.open(clahe).convert("RGB").resize((600,400))
w = A.width + B.width
h = max(A.height, B.height)
canvas = Image.new("RGB",(w,h+200),(255,255,255))
canvas.paste(A,(0,0)); canvas.paste(B,(A.width,0))
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default()
draw.text((10, h+10), "Left: Original  |  Right: CLAHE+Sauvola", fill=(0,0,0), font=font)
canvas.save("pilot_dataset/results/img10_comparison.png")

# Metin farkı (HTML)
seq = difflib.HtmlDiff().make_file(raw_txt.splitlines(), clahe_txt.splitlines(),
                                    fromdesc='Raw OCR', todesc='CLAHE OCR')
with open("pilot_dataset/results/img10_text_diff.html", "w", encoding="utf-8") as f:
    f.write(seq)

print("Kaydedildi: img10_comparison.png ve img10_text_diff.html")
