#!/usr/bin/env python3
# ЖЁСТКАЯ верификация: каждой реальной фигуре — точное число линий.
# Треугольник = 3, галка = 2, прямоугольник = 4, звезда = 5. Петля (круглая) отложена.
import glob, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
import path_extract as pe

def angle_deg(a, b, c):
    a, b, c = map(lambda p: np.array(p, float), (a, b, c))
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cosv = float(np.clip((v1 @ v2) / (n1 * n2), -1, 1))
    return math.degrees(math.acos(cosv))

def lines_of(pts, eps=5.0, merge_ang=150.0):
    """Число ПРЯМЫХ линий фигуры: RDP + склейка коллинеарных."""
    v = pe.poly_canon(pts, win=15, turn_deg=60.0, min_gap=12.0)
    closed = np.hypot(*(np.array(v[0]) - np.array(v[-1]))) < 8
    merged = v
    if closed and len(merged) > 3 and angle_deg(merged[-2], merged[0], merged[1]) > merge_ang:
        merged = merged[1:-1]                   # стык контура коллинеарен
    n = len(merged) - 1 if not closed else len(merged)
    return n, merged, closed

def main(limit=36, render=True):
    files = sorted(glob.glob('images/*.png'), key=os.path.getmtime, reverse=True)[:limit]
    rows, fails = [], []
    for f in files:
        pts, _ = pe.extract_path(f)
        if not pts or len(pts) < 8:
            continue
        n, merged, closed = lines_of(pts)
        rows.append((os.path.basename(f), n, closed, merged))
    from collections import Counter
    hist = Counter(n for _, n, _, _ in rows)
    print('распределение числа линий:', dict(sorted(hist.items())))
    print(f"{'файл':<20} линий  замкн.")
    for name, n, closed, _ in rows:
        tag = {2: 'галка/угол', 3: 'треугольник/Z', 4: 'прямоугольник/стрелка',
               5: 'звезда', 6: 'стрелка+флажок?'}.get(n, '')
        print(f"  {name:<18} {n:>3}   {'да' if closed else 'нет'}   {tag}")
    if render:
        W, H = 460, 322
        cols = 6
        rowsn = (len(rows) + cols - 1) // cols
        canvas = Image.new('RGB', (cols * W, rowsn * H), 'white')
        for j, (name, n, closed, merged) in enumerate(rows):
            f = f'images/{name}'
            im = Image.open(f).convert('RGB').resize((440, 308), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            v = [(x * 2, y * 2) for x, y in merged]
            for i in range(1, len(v)):
                d.line([v[i - 1], v[i]], fill=(220, 30, 30), width=2)
            for p in v:
                d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(255, 140, 0))
            d.rectangle([0, 0, 120, 20], fill='white')
            d.text((6, 4), f"{n} линий {'замкн' if closed else ''}", fill=(0, 0, 160))
            canvas.paste(im, ((j % cols) * W, (j // cols) * H))
        canvas.save('/tmp/lines_verify.png')
        print('рендер: /tmp/lines_verify.png')

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 36)
