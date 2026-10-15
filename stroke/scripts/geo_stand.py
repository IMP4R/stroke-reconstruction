#!/usr/bin/env python3
# GEO-STAND: эталон (скелет изображение) против ввода решателя (extract_path + HumanStroke+clamp).
# Метрики на каждую изображение: покрытие линии, средн/макс отклонение, разрывы, старт-точка, точки.
import glob, os, sys, json
import cv2
import numpy as np
import path_extract as pe
from humanize import HumanStroke

def etalon(f):
    """Скелет чернил = эталонная линия фигуры. Возвращает (skel, dot, ink)."""
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    ink = (img < 245).astype(np.uint8)
    dot = pe.find_start_dot(ink)
    ink0 = ink.copy()
    if dot:
        cv2.circle(ink0, (int(round(dot[0])), int(round(dot[1]))), 8, 0, -1)
    skel = pe.skeletonize_1px(ink0)
    return skel, dot, ink

def measure(f, seed=12345):
    skel, dot, ink = etalon(f)
    if skel is None:
        return {'file': os.path.basename(f), 'error': 'no image'}
    S = {(int(x), int(y)) for y, x in zip(*np.where(skel > 0))}
    if not S:
        return {'file': os.path.basename(f), 'error': 'empty skel'}
    pts, meta = pe.extract_path(f)
    if not pts:
        return {'file': os.path.basename(f), 'error': 'no path', 'dot': bool(dot)}
    P = np.array(pts, float)

    # 1) покрытие: доля пикселей скелета в пределах 3px от моего пути
    path_mask = np.zeros_like(skel)
    for x, y in pts:
        cv2.circle(path_mask, (int(round(x)), int(round(y))), 3, 1, -1)
    cov = (path_mask & skel).sum() / max(len(S), 1)

    # 2) отклонение моих точек от линии (дистанция до ближайшего пикселя скелета)
    Sarr = np.array(sorted(S), float)
    if len(Sarr) > 4000:
        Sarr = Sarr[np.random.RandomState(0).choice(len(Sarr), 4000, replace=False)]
    devs = []
    for x, y in P[:: max(1, len(P) // 200)]:
        d = np.sqrt(((Sarr - [x, y]) ** 2).sum(1)).min()
        devs.append(d)
    devs = np.array(devs)

    # 3) разрывы: непрерывность пути (шаги точек)
    steps = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1))
    big_jumps = int((steps > 15).sum())

    # 4) старт
    d0 = ((P[0][0] - dot[0]) ** 2 + (P[0][1] - dot[1]) ** 2) ** 0.5 if dot else -1

    # 5) HumanStroke+clamp → выходная траектория Q, её покрытие и отклонение
    hs = HumanStroke(seed=seed)
    evs = hs.generate(pts)
    Q = np.array([[e[1] * 220 / 100, e[2] * 154 / 100] for e in evs if e[0] == 'mousemove'])
    qmask = np.zeros_like(skel)
    for x, y in Q:
        cv2.circle(qmask, (int(round(x)), int(round(y))), 3, 1, -1)
    cov_q = (qmask & skel).sum() / max(len(S), 1)
    qdev = []
    for x, y in Q[:: max(1, len(Q) // 200)]:
        qdev.append(np.sqrt(((Sarr - [x, y]) ** 2).sum(1)).min())
    qdev = np.array(qdev)

    # off-line: доля точек штриха дальше 3.5px от чернил (хорды/прыжки)
    ink_mask = ink if ink is not None else (skel > 0).astype(np.uint8)
    if len(ink_mask.shape) == 2:
        big = cv2.dilate(ink_mask, np.ones((7, 7), np.uint8))
        off = 0
        for x, y in Q:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < big.shape[0] and 0 <= xi < big.shape[1] and big[yi, xi] == 0:
                off += 1
        offline_frac = off / max(len(Q), 1)

    return {'file': os.path.basename(f), 'cov_ext': round(float(cov), 3),
            'offline_frac': round(float(offline_frac), 3),
            'cov_stroke': round(float(cov_q), 3),
            'dev_ext_mean': round(float(devs.mean()), 2), 'dev_ext_max': round(float(devs.max()), 1),
            'dev_stroke_max': round(float(qdev.max()), 1),
            'jumps': big_jumps, 'start_dot_px': round(float(d0), 1),
            'n_ext': len(pts), 'n_stroke': len(Q)}

def main():
    files = sorted(glob.glob('images/*.png'), key=os.path.getmtime, reverse=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'all':
        pass
    else:
        files = files[:60]
    rows = []
    for f in files:
        try:
            r = measure(f)
            rows.append(r)
        except Exception as e:
            rows.append({'file': os.path.basename(f), 'error': str(e)[:60]})
    ok = [r for r in rows if 'error' not in r]
    print(f"проверено {len(rows)}, с ошибками {len(rows)-len(ok)}")
    if ok:
        worst_cov = sorted(ok, key=lambda r: r['cov_stroke'])[:10]
        worst_dev = sorted(ok, key=lambda r: -r['offline_frac'])[:8]
        print("\n=== ХУДШЕЕ ПОКРЫТИЕ ШТРИХА (cov_stroke) ===")
        for r in worst_cov:
            print(f"  {r['file']:<18} cov_ext={r['cov_ext']} cov_stroke={r['cov_stroke']} dev_max={r['dev_stroke_max']} jumps={r['jumps']} n={r['n_ext']}→{r['n_stroke']}")
        print("\n=== ХУДШАЯ ДОЛЯ ВНЕ ЛИНИИ (хорды) ===")
        for r in worst_dev:
            print(f"  {r['file']:<18} offline={r['offline_frac']} cov_stroke={r['cov_stroke']} dev_max={r['dev_stroke_max']} n={r['n_stroke']}")
        import statistics as st
        print(f"\nмедианы: cov_ext={st.median([r['cov_ext'] for r in ok])} cov_stroke={st.median([r['cov_stroke'] for r in ok])} "
              f"offline={st.median([r['offline_frac'] for r in ok])} "
              f"dev_ext_mean={st.median([r['dev_ext_mean'] for r in ok])} jumps={st.median([r['jumps'] for r in ok])} "
              f"start_dot={st.median([r['start_dot_px'] for r in ok])}")
    errs = [r for r in rows if 'error' in r]
    if errs:
        print("\nошибки:", Counter := {})
        for r in errs:
            Counter[r['error']] = Counter.get(r['error'], 0) + 1

if __name__ == '__main__':
    main()
