"""Извлечение траектории обводки из PNG-изображение image (220x154, серая линия + стартовая точка)."""
import cv2
import numpy as np


def find_start_dot(ink: np.ndarray) -> tuple[int, int]:
    """Стартовая точка — диск ~5-7px радиуса, толще любой линии.

    distanceTransform: пиксели с r>=3.5 есть только внутри диска (линия тоньше).
    Из кандидатов берём самый крупный круглый блоб; если на 3.5 пусто — снижаем порог.
    """
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
    for thr in (3.5, 3.0, 2.5, 2.0):
        mask = (dt >= thr).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        best = None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 3:
                continue
            ys, xs = np.where(labels == i)
            cx, cy = float(xs.mean()), float(ys.mean())
            r = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean()))
            if r < 1.2:
                continue
            circ = np.pi * r * r / area  # ~1.0 у диска, >>1 у вытянутых
            if circ > 1.8:
                continue
            score = area / circ
            if best is None or score > best[0]:
                best = (score, (cx, cy))
        if best is not None:
            return best[1]
    return None


def thin(ink: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning на numpy (без ximgproc)."""
    img = (ink > 0).astype(np.uint8)
    changed = True
    P = lambda im, x, y: im[y, x]
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1)
            p2 = padded[:-2, 1:-1]; p3 = padded[:-2, 2:]; p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]; p6 = padded[2:, 1:-1]; p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]; p9 = padded[:-2, :-2]
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum((seq[i] == 0) & (seq[i + 1] == 1) for i in range(8))
            if step == 0:
                cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if cond.any():
                img[cond] = 0
                changed = True
    return img


def skeletonize_1px(ink, spur_max=10):
    """Скелет в 1px: skimage.skeletonize + обрезка коротких шипов + склейка разрывов."""
    from skimage.morphology import skeletonize
    sk = skeletonize(ink > 0).astype(np.uint8)

    def snap():
        ys, xs = np.where(sk > 0)
        pts = set(zip(xs.tolist(), ys.tolist()))
        def nbrs8(p):
            x, y = p
            return sum(((x+dx, y+dy) in pts) for dx in (-1,0,1) for dy in (-1,0,1)) - 1
        return pts, nbrs8

    # 1) обрезка шипов: семантика старой версии (шип = ветка от кончика,
    # НЕ доходящая до перекрёстка, короче spur_max), но множество пикселей
    # строится ОДИН раз и обновляется инкрементально — было ~25мс, стало ~2мс
    ys, xs = np.where(sk > 0)
    pts = set(zip(xs.tolist(), ys.tolist()))

    def nbrs8(p):
        x, y = p
        return [q for q in ((x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
                if q in pts]

    for _ in range(60):
        tips = [p for p in pts if len(nbrs8(p)) == 1]
        removed = False
        for t in tips:
            if t not in pts:
                continue
            path = [t]
            seen = {t}
            cur = t
            hit = False
            while True:
                nb = [q for q in nbrs8(cur) if q not in seen]
                if not nb:
                    break
                cur = nb[0]
                seen.add(cur)
                path.append(cur)
                if len(nbrs8(cur)) >= 3:
                    hit = True
                    break
            if not hit and len(path) <= spur_max:
                pts.difference_update(path)
                for (x, y) in path:
                    sk[y, x] = 0
                removed = True
        if not removed:
            break

    # 2) склейка близких кончиков (разрывы после обрезки/утоньшения)
    pts, nbrs8 = snap()
    ends = [p for p in pts if nbrs8(p) == 1]
    for i, (x1, y1) in enumerate(ends):
        for (x2, y2) in ends[i+1:]:
            if 1 <= np.hypot(x1-x2, y1-y2) <= 5:
                cv2.line(sk, (x1, y1), (x2, y2), 1, 1)
    return sk


def route_on_ink(ink, a, b, dilate=True):
    """Кратчайший путь по чернилам (BFS 8-связность) от a до b; деградация — прямая."""
    h, w = ink.shape
    ax, ay = int(round(a[0])), int(round(a[1]))
    bx, by = int(round(b[0])), int(round(b[1]))

    def free(x, y):
        return 0 <= x < w and 0 <= y < h and ink[y, x]

    if dilate:
        # всегда чуть расширяем маску: сглаживание рвёт сужающиеся концы линий
        # на изолированные песчинки, и BFS без дилатации обходит всё кольцо
        ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))

    from collections import deque
    prev = {(ax, ay): None}
    q = deque([(ax, ay)])
    goal = (bx, by)
    found = False
    while q:
        cur = q.popleft()
        if cur == goal:
            found = True
            break
        x, y = cur
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if nb not in prev and free(nb[0], nb[1]):
                    prev[nb] = cur
                    q.append(nb)
    if not found:
        for t in np.linspace(0, 1, max(int(np.hypot(b[0]-a[0], b[1]-a[1]) / 2), 2))[1:]:
            yield (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)
        return
    # восстановить путь
    pts = []
    cur = goal
    while cur is not None:
        pts.append(cur)
        cur = prev[cur]
    pts.reverse()
    # проредить до ~2px
    out = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0]-out[-1][0], p[1]-out[-1][1]) >= 2:
            out.append(p)
    for p in out[1:]:
        yield p


def _same(a, b):
    """Безопасное сравнение id узлов (int/tuple/None)."""
    if a is None or b is None:
        return a is None and b is None
    try:
        return bool(a == b)
    except ValueError:
        return False


def order_chain_px(pxs):
    """Порядок пикселей цепочки: BFS от концевого (степень<=1) пикселя.

    Для цепи-пути BFS-дистанция монотонна вдоль дуги — порядок точный,
    без карманов и повторов (в отличие от NN-обхода, срезающего углы).
    """
    from collections import deque
    pxs = set(pxs)

    def nbrs8(p):
        x, y = p
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and (x + dx, y + dy) in pxs]

    ends = [p for p in pxs if len(nbrs8(p)) <= 1]
    start = ends[0] if ends else sorted(pxs)[0]
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in nbrs8(cur):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    order = sorted(pxs, key=lambda p: dist.get(p, 1 << 30))
    return np.array(order, float)


def extract_path(png_path: str, min_points: int = 24, max_points: int = 90):
    """PNG -> [(x,y)] в пикселях холста, от стартовой точки до конца штриха.

    Рёберная модель: перекрёстки = кластеры узлов, рёбра = цепочки между узлами.
    Каждый узел проходим сколько угодно раз, каждое ребро — ровно один раз;
    на развилке приоритет — продолжать в том же направлении (правило пентаграммы).
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    ink = (img < 245).astype(np.uint8)
    start = find_start_dot(ink)
    # уточнение: точка сидит на КОНЦЕ штриха. Если под блобом скелет не кончается —
    # это толстый угол/перекрёсток; берём ближайший круглый блоб у кончика степени 1.
    def _dot_on_tip(cand):
        if cand is None:
            return None
        sk = skeletonize_1px((ink > 0).astype(np.uint8))
        ys, xs = np.where(sk > 0)
        pts_s = set(zip(xs.tolist(), ys.tolist()))
        def deg(p):
            x, y = p
            return sum(((x+dx, y+dy) in pts_s) for dx in (-1,0,1) for dy in (-1,0,1)) - 1
        cx, cy = int(round(cand[0])), int(round(cand[1]))
        for r in range(0, 10):
            near = [(x, y) for x in range(cx-r, cx+r+1) for y in range(cy-r, cy+r+1)
                    if (x, y) in pts_s]
            if any(deg(p) <= 1 for p in near):
                return cand
        return None
    if _dot_on_tip(start) is None:
        # ищем альтернативные круглые блобы на кончиках
        dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        sk = skeletonize_1px((ink > 0).astype(np.uint8))
        ys, xs = np.where(sk > 0)
        pts_s = set(zip(xs.tolist(), ys.tolist()))
        def deg(p):
            x, y = p
            return sum(((x+dx, y+dy) in pts_s) for dx in (-1,0,1) for dy in (-1,0,1)) - 1
        tips = [p for p in pts_s if deg(p) == 1]
        best = None
        for (tx, ty) in tips:
            r = dt[ty, tx]
            if r < 1.6:
                continue
            score = r
            if best is None or score > best[0]:
                best = (score, (float(tx), float(ty)))
        if best is not None:
            start = best[1]
    if start is None:
        return None, None
    # диск стартовой точки убираем — иначе он режет линию и ломает скелет
    ink0 = ink.copy()
    cv2.circle(ink0, (int(round(start[0])), int(round(start[1]))), 8, 0, -1)
    # СМЫКАНИЕ: перекрёстки двух проходов не касаются в скелете —
    # морфологическое закрытие сращивает их до прореживания
    ink0 = cv2.morphologyEx(ink0, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    skel = skeletonize_1px(ink0)
    ys, xs = np.where(skel > 0)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return None, None

    def deg(p):
        x, y = p
        return sum(((x + dx, y + dy) in pts) for dx in (-1, 0, 1) for dy in (-1, 0, 1)) - 1

    # УЗЕЛ = истинный X-перекрёсток (deg>=4). Зоны двойного штриха дают
    # параллельные скелетные линии (deg=3) — это НЕ узлы, ребро идёт сквозь.
    junction_px = {p for p in pts if deg(p) >= 3}
    # кластеризацияdeg>=3 пикселей со сцепкой 2.5px (ступенчатые T одного
    # перекрёстка) БЕЗ дилатации — толстые зоны не захватываются
    jp = list(junction_px)
    parent = list(range(len(jp)))
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i in range(len(jp)):
        for j in range(i + 1, len(jp)):
            if abs(jp[i][0]-jp[j][0]) <= 2 and abs(jp[i][1]-jp[j][1]) <= 2:
                parent[find(i)] = find(j)
    jmask = np.zeros_like(skel)
    jlab = np.zeros(len(jp), int)
    lab = np.zeros_like(skel)
    for i, (x, y) in enumerate(jp):
        jmask[y, x] = 1
        lab[y, x] = find(i) + 1
    nn = len(set(find(i) for i in range(len(jp)))) + 1
    node_id = {}
    nodes = {}
    for y, x in zip(*np.where(lab > 0)):
        # только те пиксели, что реально узловые (дилатация — для склейки кластеров)
        if (x, y) in junction_px or skel[y, x]:
            c = int(lab[y, x]) - 1
            node_id[(int(x), int(y))] = c
            nodes.setdefault(c, []).append((int(x), int(y)))
    # цепочки = компоненты связности не-узловой части
    cmask = np.zeros_like(skel)
    for (x, y) in pts:
        if (x, y) not in node_id:
            cmask[y, x] = 1
    nc, clab = cv2.connectedComponents(cmask, connectivity=8)
    chains = {}
    for y, x in zip(*np.where(clab > 0)):
        chains.setdefault(clab[y, x], []).append((int(x), int(y)))

    # рёбра: цепочка + узлы на концах (соседство в радиусе 2)
    edges = []
    node_edges = {c: [] for c in nodes}
    for cid_, pxs in chains.items():
        pxset = set(pxs)
        ordered_px = order_chain_px(set(pxs))
        ends_px = ordered_px[:5] + ordered_px[-5:]   # только концы цепочки
        adj_nodes = set()
        for (x, y) in ends_px:
            for ddx in range(-4, 5):
                for ddy in range(-4, 5):
                    q = (x + ddx, y + ddy)
                    if q in node_id:
                        adj_nodes.add(node_id[q])
        adj_nodes = sorted(adj_nodes)
        a = adj_nodes[0] if adj_nodes else None
        b = adj_nodes[1] if len(adj_nodes) > 1 else None
        i = len(edges)
        edges.append([set(pxs), a, b, order_chain_px(set(pxs)), None])
        if a is not None:
            node_edges[a].append(i)
        if b is not None and not _same(a, b):
            node_edges[b].append(i)

    # КОНТРАКТ МИКРО-ЦЕПОЧЕК: цепочка <2.5px между двумя узлами = артефакт
    # перекрёста. Узлы концов сливаются в один кластер (перекрёсток один),
    # связность живёт в узле — микро-ребро исчезает, не рвя граф.
    _changed = True
    while _changed:
        _changed = False
        for _i in range(len(edges)):
            _pxs, _a, _b, _ord, _ = edges[_i]
            if _a is None or _b is None or _same(_a, _b):
                continue
            _arr = np.array(_ord, float)
            if float(np.linalg.norm(_arr[-1] - _arr[0])) < 2.5:
                for _e2 in edges:
                    if _e2[1] is not None and _same(_e2[1], _b):
                        _e2[1] = _a
                    if _e2[2] is not None and _same(_e2[2], _b):
                        _e2[2] = _a
                nodes[_a] = nodes.get(_a, []) + nodes.pop(_b, [])
                _merged = [x for x in dict.fromkeys(
                    node_edges.get(_a, []) + node_edges.pop(_b, [])) if x != _i]
                node_edges[_a] = _merged
                edges[_i] = [set(), None, None, [(0.0, 0.0)], None]
                _changed = True
                break
    # полная пересборка списка рёбер узлов (контракт менял ключи)
    node_edges = {}
    for _i2, _e2 in enumerate(edges):
        _a2, _b2 = _e2[1], _e2[2]
        if _a2 is None and _b2 is None:
            continue
        _arr2 = np.array(_e2[3], float)
        if float(np.linalg.norm(_arr2[-1] - _arr2[0])) < 1.5:
            continue
        if _a2 is not None:
            node_edges.setdefault(_a2, []).append(_i2)
        if _b2 is not None and not _same(_a2, _b2):
            node_edges.setdefault(_b2, []).append(_i2)

    # средняя толщина (distanceTransform по СЫРЫМ чернилам) каждого ребра:
    # толстые рёбра (mean_dt >= 2.6) генератор проводил дважды — «для обхода»
    _dt_raw = cv2.distanceTransform(ink, cv2.DIST_L2, 5)

    def edge_mean_dt(i):
        if edges[i][4] is None:
            pxs_i = edges[i][0]
            vals = [_dt_raw[y, x] for x, y in pxs_i]
            edges[i][4] = float(np.mean(vals)) if vals else 0.0
        return edges[i][4]

    sx, sy = start

    # мостикаем свободные концы (разрез диска, дырки скелета): жадная сварка
    # ближайших встречных концов в радиусе 22px
    def edge_pts_array(i):
        return np.array(sorted(edges[i][0]), float)

    def loose_ends():
        out = []
        for i in range(len(edges)):
            pxs, a, b, arr, _dtm = edges[i]
            if a is None or b is None:
                p1, p2 = arr[0], arr[-1]
                dirv = (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-9)
                if a is None:
                    out.append({'e': i, 'side': 1, 'p': p1, 'dir': dirv})
                if b is None:
                    out.append({'e': i, 'side': 2, 'p': p2, 'dir': -dirv})
        return out

    while True:
        ends = [e for e in loose_ends()
                if edges[e['e']][e['side']] is None]
        best = None
        for m in range(len(ends)):
            for n in range(m + 1, len(ends)):
                e1, e2 = ends[m], ends[n]
                if e1['e'] == e2['e'] and e1['side'] == e2['side']:
                    continue
                dist = float(np.hypot(*(e1['p'] - e2['p'])))
                if dist > 22:
                    continue
                # концы должны смотреть друг на друга
                if float(e1['dir'] @ e2['dir']) > -0.2:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, e1, e2)
        if best is None:
            break
        _, e1, e2 = best
        syn = ('syn', e1['e'], e1['side'], e2['e'], e2['side'])
        nodes[syn] = [tuple(e1['p']), tuple(e2['p'])]
        edges[e1['e']][e1['side']] = syn
        edges[e2['e']][e2['side']] = syn
        node_edges[syn] = []
    for i, edg in enumerate(edges):
        a, b = edg[1], edg[2]
        if a is not None:
            node_edges.setdefault(a, []).append(i)
        if b is not None and not _same(a, b):
            node_edges.setdefault(b, []).append(i)

    def edge_endpoints(i):
        arr = edges[i][3]
        return arr[0], arr[-1]

    def edge_dir_at(i, from_node):
        """Направление выхода из узла на ребро i."""
        p1, p2 = edge_endpoints(i)
        np_node = np.array(nodes[from_node]).mean(0)
        return p2 - p1 if np.linalg.norm(p1 - np_node) < np.linalg.norm(p2 - np_node) else p1 - p2

    # ФЛЁРИ с мостами: рёбра выписываются В ПОРЯДКЕ обхода (никакой реверс-
    # эмиссии). Мост проходим только когда другого выхода нет. На развилке:
    # сквозь пересечение (прямо) > двойной штрих (rem=2) > прямо.
    rem = {i: (2 if edge_mean_dt(i) >= 2.6 else 1) for i in range(len(edges))}
    start_edge = min(range(len(edges)),
                     key=lambda i: min(float(np.linalg.norm(np.array(edges[i][3][0]) - [sx, sy])),
                                       float(np.linalg.norm(np.array(edges[i][3][-1]) - [sx, sy]))))
    arr0 = np.array(edges[start_edge][3], float)
    near0 = np.linalg.norm(arr0[0] - [sx, sy]) <= np.linalg.norm(arr0[-1] - [sx, sy])
    path = [(float(sx), float(sy))]
    global _TRACE
    _TRACE = []
    cur = edges[start_edge][1] if near0 else edges[start_edge][2]
    o0 = arr0 if near0 else arr0[::-1]
    path.extend((float(x), float(y)) for x, y in o0)   # стартовая цепочка (пиксели)
    rem[start_edge] -= 1
    _TRACE.append(('go', start_edge, 0.0, rem[start_edge]))
    tail = np.array(path[-1]) - np.array(path[max(0, len(path) - 8)])
    cur_dir = tail / max(float(np.linalg.norm(tail)), 1e-9)
    guard = 0
    total_left = lambda: sum(rem.values())
    def reachable_from(nd, skip=None):
        if nd is None:
            return 0
        seen, stack_ = set(), [nd]
        left = {i: c for i, c in rem.items() if c > 0 and i != skip}
        while stack_:
            x = stack_.pop()
            for e2 in node_edges.get(x, []):
                if left.get(e2, 0) > 0:
                    left.pop(e2, None)
                    a2, b2 = edges[e2][1], edges[e2][2]
                    nxt = b2 if (a2 is not None and _same(a2, x)) else a2
                    if nxt is not None and nxt not in seen:
                        seen.add(nxt)
                        stack_.append(nxt)
        return len(left)
    import os as _os
    if _os.environ.get('GEO_DEBUG'):
        print(f"DBG cur={cur!r} тип={type(cur)} в_узлах={cur in nodes if cur else '-'} "
              f"в_словаре={cur in node_edges if cur else '-'} осталось={total_left()}")
    ink_bin = (ink > 0).astype(np.uint8) if ink is not None else (ink0 > 0).astype(np.uint8)
    while total_left() > 0 and guard < 500:
        guard += 1
        pen = np.array(path[-1], float)
        # ПРИОРИТЕТ ПЕРЕКРЁСТКА: перо идёт по чернилам. Кандидат — неиспользо-
        # ванное ребро в радиусе 16px; иначе — ближайшее ДОСТИЖИМОЕ ПО ЧЕРНИЛАМ
        # (это и есть проход сквозь перекрёсток, без хорд по пустоте).
        cand = []
        for i in range(len(edges)):
            if rem[i] <= 0:
                continue
            arr = np.array(edges[i][3], float)
            if min(float(np.linalg.norm(arr[0] - pen)),
                   float(np.linalg.norm(arr[-1] - pen))) <= 20.0:
                cand.append(i)
        bridge = []
        if not cand:
            best_i, best_n, best_rt = None, 1e18, None
            for i in range(len(edges)):
                if rem[i] <= 0:
                    continue
                arr = np.array(edges[i][3], float)
                for pt in (arr[0], arr[-1]):
                    rt = list(route_on_ink(ink_bin, pen, pt))
                    if rt and len(rt) < best_n:
                        best_n, best_i, best_rt = len(rt), i, rt
            if best_i is None:
                break                      # чернила кончились — конец штриха
            path.extend((float(x), float(y)) for x, y in best_rt)
            cand = [best_i]
        def sc(i):
            arr = np.array(edges[i][3], float)
            d = (arr[-1] - arr[0]) if (np.linalg.norm(arr[0] - pen) <=
                                       np.linalg.norm(arr[-1] - pen)) else (arr[0] - arr[-1])
            n = float(np.linalg.norm(d))
            return float(d @ cur_dir) / n if (n > 1e-9 and cur_dir is not None) else 0.0
        st_all = {i: sc(i) for i in cand}
        # ПРИОРИТЕТ — ПРЯМАЯ (ребро двунаправленное, идём куда смотрит перо);
        # двойной штрих — лишь тай-брейк при равной прямоте
        pick = max(cand, key=lambda i: (round(st_all[i], 2), rem[i]))
        rem[pick] -= 1
        _TRACE.append(('go', pick, round(st_all[pick], 2), rem[pick]))
        arr = np.array(edges[pick][3], float)
        o = arr if (np.linalg.norm(arr[0] - np.array(path[-1])) <=
                    np.linalg.norm(arr[-1] - np.array(path[-1]))) else arr[::-1]
        path.extend((float(x), float(y)) for x, y in o)
        tail = np.array(path[-1]) - np.array(path[max(0, len(path) - 8)])
        n = float(np.linalg.norm(tail))
        if n > 1e-9:
            cur_dir = tail / n
    # ФИНАЛЬНАЯ СБОРКА: старт строго в точке; КАЖДЫЙ разрыв >3px зашиваем
    # маршрутом по чернилам (никаких линий по пустому месту); замкнутые
    # фигуры явно доводим обратно до точки старта.
    # замкнутые фигуры НЕ доводим обратно в точку: штрих заканчивается там,
    # где закончился периметр (последний вырезанный конец рядом с точкой)
    raw_ink = (ink > 0).astype(np.uint8) if ink is not None else ink0
    final = [path[0]]
    for p in path[1:]:
        a = np.array(final[-1], float)
        b = np.array(p, float)
        dist_ab = float(np.linalg.norm(b - a))
        if dist_ab > 3.0:
            brg = route_on_ink(raw_ink, a, b)
            if brg:
                final.extend((float(px), float(py)) for px, py in brg)
            elif dist_ab > 8.0:
                break        # обрыв — штрих заканчивается здесь, хвост отбрасываем
        final.append((float(p[0]), float(p[1])))
    global _LAST_WALK, _LAST_EDGES
    _LAST_WALK = list(final)
    _LAST_EDGES = [(i, (float(e[3][0][0]), float(e[3][0][1])), (float(e[3][-1][0]), float(e[3][-1][1])), str(e[1]), str(e[2]), edge_mean_dt(i)) for i, e in enumerate(edges)]
    return resample(final, min_points, max_points), (int(round(sx)), int(round(sy)))


def resample(path, min_points, max_points):
    if len(path) < 3:
        return path or None
    arr = np.array(path, dtype=float)
    seg = np.sqrt(((arr[1:] - arr[:-1]) ** 2).sum(axis=1))
    keep = seg > 0
    if not keep.any():
        return path
    arr = np.concatenate([arr[:-1][keep], arr[-1:]], 0)
    seg = seg[keep]
    total = seg.sum()
    cum = np.concatenate([[0], np.cumsum(seg)])
    npts = int(np.clip(total / 2.2, min_points, max_points))
    want = np.linspace(0, total, npts + 1)
    out = [arr[0]]
    j = 0
    for w in want[1:]:
        while j < len(seg) and cum[j + 1] < w:
            j += 1
        if j >= len(seg):
            out.append(arr[-1]); continue
        tt = (w - cum[j]) / max(seg[j], 1e-9)
        out.append(arr[j] * (1 - tt) + arr[j + 1] * tt)
    res = [(float(x), float(y)) for x, y in out]
    return res


if __name__ == '__main__':
    import sys
    path, start = extract_path(sys.argv[1])
    print('start dot:', start, 'points:', len(path) if path else 0)
    if path:
        print('first 5:', [(round(x), round(y)) for x, y in path[:5]])
        print('last 5:', [(round(x), round(y)) for x, y in path[-5:]])


def rdp(points, eps=5.0):
    """Рамер-Дуглас-Пекер: ломаная только из углов (треугольник = 3 отрезка)."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 3:
        return pts
    def _rdp(lo, hi):
        if hi <= lo + 1:
            return [lo, hi]
        a, b = np.array(pts[lo]), np.array(pts[hi])
        ab = b - a
        ab2 = float(ab @ ab) or 1e-9
        dmax, imax = -1.0, lo
        for k in range(lo + 1, hi):
            t = float((np.array(pts[k]) - a) @ ab) / ab2
            t = min(max(t, 0.0), 1.0)
            d = float(np.linalg.norm(np.array(pts[k]) - (a + ab * t)))
            if d > dmax:
                dmax, imax = d, k
        if dmax > eps:
            left = _rdp(lo, imax)
            right = _rdp(imax, hi)
            return left[:-1] + right
        return [lo, hi]
    keep = sorted(set(_rdp(0, len(pts) - 1)))
    return [pts[k] for k in keep]


def poly_canon(points, win=9, turn_deg=40.0, min_gap=8.0):
    """Ломаная по кривизне: угол только на настоящем переломе (поворот
    >turn_deg), между углами — прямые. Дрожь линии углов не рождает."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < win * 2:
        return pts
    # равномерная передискретизация по 2px для стабильных касательных
    res = [pts[0]]
    acc = 0.0
    for p, q in zip(pts, pts[1:]):
        d = ((q[0]-p[0])**2 + (q[1]-p[1])**2) ** 0.5
        acc += d
        if acc >= 2.0:
            res.append(q)
            acc = 0.0
    if res[-1] != pts[-1]:
        res.append(pts[-1])
    P = np.array(res, float)
    h = win // 2
    turns = []
    for i in range(h, len(P) - h):
        v1 = P[i] - P[i - h]
        v2 = P[i + h] - P[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            turns.append(0.0); continue
        cosv = float(np.clip((v1 @ v2) / (n1 * n2), -1, 1))
        turns.append(math_deg := np.degrees(np.arccos(cosv)))
    corners = [0]
    for k, t in enumerate(turns):
        if t >= turn_deg:
            i = k + h
            if i - corners[-1] >= min_gap:
                corners.append(i)
    if len(P) - 1 - corners[-1] >= min_gap:
        corners.append(len(P) - 1)
    elif corners[-1] != len(P) - 1:
        corners[-1] = len(P) - 1
    return [tuple(P[i]) for i in corners]


def stroke_corners(png_path):
    """Простейшая ломаная: углы = вершины выпуклой оболочки канон-полилинии
    (+ старт в точке), порядок = по чернилам от точки. Треугольник = 3 линии."""
    pts, start = extract_path(png_path)
    if not pts or len(pts) < 6:
        return None, start
    import cv2
    canon = poly_canon(pts, win=11, turn_deg=48.0)
    P = np.array(canon, float)
    hull = cv2.convexHull(P.astype(np.float32)).reshape(-1, 2)
    # углы на оболочке (в пределах 7px)
    keep = [tuple(p) for p in P if min(float(np.linalg.norm(p - h)) for h in hull) <= 7.0]
    # дедуп
    ded = []
    for p in keep:
        if all(float(np.hypot(p[0]-q[0], p[1]-q[1])) > 10.0 for q in ded):
            ded.append(p)
    if start is not None:
        ded.append((float(start[0]), float(start[1])))
    # порядок: жадно по длине пути ПО ЧЕРНИЛАМ
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    ink_bin = (img < 245).astype(np.uint8)
    ordered = [ded[-1]]          # старт из точки
    rest = [p for p in ded[:-1] if float(np.hypot(p[0]-ded[-1][0], p[1]-ded[-1][1])) > 10.0]
    _dtmap = cv2.distanceTransform((img < 245).astype(np.uint8), cv2.DIST_L2, 5)
    covered = np.zeros_like(_dtmap, bool)          # пройденные чернила
    while rest:
        pen = np.array(ordered[-1], float)
        best, bl, best_key, bl_dev = None, 1e18, None, 1e18
        for p in rest:
            q = np.array(p, float)
            straight = float(np.hypot(p[0]-pen[0], p[1]-pen[1]))
            rt = list(route_on_ink(ink_bin, pen, q))
            if not rt:
                continue
            L = len(rt) * 2.0
            if L > straight * 1.35 + 6.0:
                continue            # прямой связи по чернилам нет
            # НЕПРЕРЫВНОСТЬ ПЕРА: касательная маршрута в первые 12px против
            # направления ПРЕДЫДУЩЕГО хода. Свой путь — мягкое продолжение,
            # чужой (через перекрёсток на другую линию) — резкий излом.
            h = max(min(6, len(rt) - 1), 1)
            v0 = np.array(rt[h], float) - np.array(rt[0], float)
            seg = q - pen
            ref = seg if len(ordered) < 2 else np.array(ordered[-1], float) - np.array(ordered[-2], float)
            n0 = float(np.hypot(*v0)) + 1e-9
            n1 = float(np.hypot(*ref)) + 1e-9
            turn = float(np.degrees(np.arccos(np.clip((v0 @ ref) / (n0 * n1), -1, 1))))
            dsv = [_dtmap[min(max(int(round(pt[1])), 0), _dtmap.shape[0]-1),
                            min(max(int(round(pt[0])), 0), _dtmap.shape[1]-1)] for pt in rt]
            thickc = (min(dsv) >= 2.2) if len(dsv) >= 4 else False
            # СТРОГО ПО МЕНЬШЕМУ УГЛУ (тай-брейк — длина по чернилам)
            key = (round(turn, 1), round(L, 1))
            if best_key is None or key < best_key:
                best, bl, best_key = p, L, key
        if best is None:
            break                   # штрих заканчивается
        q = np.array(best, float)
        # ТОЛСТАЯ нога: чернильный маршрут ЦЕЛИКОМ в двойном штрихе
        rt3 = list(route_on_ink(ink_bin, pen, q))
        ds = []
        for pt in rt3:
            xi, yi = int(round(pt[0])), int(round(pt[1]))
            if 0 <= yi < _dtmap.shape[0] and 0 <= xi < _dtmap.shape[1]:
                ds.append(_dtmap[yi, xi])
        thick = (min(ds) >= 2.2) if len(ds) >= 4 else False
        for pt in rt3:
            covered[min(max(int(round(pt[1])), 0), covered.shape[0]-1),
                    min(max(int(round(pt[0])), 0), covered.shape[1]-1)] = True
        covered[:] = cv2.dilate(covered.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        ordered.append(best)
        if thick:
            ordered.append(tuple(pen))   # возврат пера: ретрейс
        else:
            # маршрут лёг через УЖЕ пройденный угол — проход через него явный
            rt2 = list(route_on_ink(ink_bin, pen, q))
            if rt2:
                mid = rt2[len(rt2) // 2]
                for k, p0 in enumerate(ordered[:-2]):
                    if float(np.hypot(mid[0]-p0[0], mid[1]-p0[1])) <= 10.0:
                        ordered.insert(len(ordered) - 1, ordered[k])
                        break
        rest.remove(best)
    # ПРУНИНГ по ГОТОВОМУ маршруту: выброс руки на прямой (разворот <25°)
    if len(ordered) > 3:
        changed = True
        while changed:
            changed = False
            for i in range(1, len(ordered) - 1):
                a, b, c = (np.array(ordered[i-1], float), np.array(ordered[i], float),
                           np.array(ordered[i+1], float))
                v1, v2 = a - b, c - b
                n1 = float(np.hypot(*v1)) + 1e-9
                n2 = float(np.hypot(*v2)) + 1e-9
                turn = float(np.degrees(np.arccos(float(np.clip((v1 @ v2) / (n1 * n2), -1, 1)))))
                if turn < 25.0:
                    del ordered[i]
                    changed = True
                    break
    # схлопнуть подряд идущие дубли (артефакт ретрейса)
    dedup = []
    for p in ordered:
        if not dedup or float(np.hypot(p[0]-dedup[-1][0], p[1]-dedup[-1][1])) > 12.0:
            dedup.append(p)
    ordered = dedup

    # ЗАМЫКАЮЩАЯ СТОРОНА: на хорде последний-угол→точка лежат СВЕЖИЕ
    # (нарисованные, но ещё не пройденные) чернила — генератор возвращается
    # в точку по остатку стороны. Фантом идёт по пройденным — отказ.
    if len(ordered) >= 3:
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        _ink = (img < 245).astype(np.uint8)
        _fresh = _ink & ~cv2.dilate(covered.astype(np.uint8), np.ones((3, 3), np.uint8))
        _freshd = cv2.dilate(_fresh, np.ones((7, 7), np.uint8)) > 0
        _dot = np.array(ordered[0], float)
        _last = np.array(ordered[-1], float)
        _straight = float(np.hypot(*(_dot - _last)))
        n_s = max(int(_straight / 4.0), 3)
        cov = 0
        for k in range(n_s + 1):
            t = k / n_s
            x, y = _last * (1-t) + _dot * t
            xi, yi = min(max(int(round(x)), 0), _freshd.shape[1]-1), min(max(int(round(y)), 0), _freshd.shape[0]-1)
            if _freshd[yi, xi]:
                cov += 1
        if cov / (n_s + 1) >= 0.75:
            ordered.append((_dot[0], _dot[1]))

    return ordered, start
