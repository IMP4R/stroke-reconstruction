from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
out = root / 'fresh_output'
out.mkdir(exist_ok=True)
files = sorted((root / 'images').glob('*.png'))
for batch in range(4):
    sheet = Image.new('RGB', (1100, 880), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(files[batch*25:(batch+1)*25]):
        x, y = i % 5 * 220, i // 5 * 176
        sheet.paste(Image.open(f).convert('RGB'), (x, y+20))
        draw.text((x+8,y+3), f.stem, fill='black')
    sheet.save(out / f'inputs_{batch+1}.png')
