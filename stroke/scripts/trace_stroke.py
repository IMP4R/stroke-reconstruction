"""Continuous-ink centerline tracing. Independent of the legacy extractor.
No closing, endpoint snapping, convex hulls, Euler tours or class templates.
Coordinates are image x,y. Dense route is authoritative; SVG is ink-constrained.
"""
from __future__ import annotations
import argparse
import json
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def prepare(image):
    gray = np.asarray(image.convert('L'), dtype=float)
    darkness = 255 - gray
    ink_peak = float(darkness.max())
    if ink_peak < 1:
        raise ValueError('No ink')
    mask = darkness >= max(1, ink_peak * .22)
    distance = ndi.distance_transform_edt(mask)

    # Search all meaningful local DT maxima.  On thin drawings the start dot can
    # be narrower than a repeatedly drawn arrow wing or corner.
    maxima = distance == ndi.maximum_filter(distance, size=5, mode='constant')
    cores, count = ndi.label(maxima & (distance >= 1.4), np.ones((3, 3)))
    skeleton = skeletonize(mask)
    degree = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant') - skeleton
    endpoint_y, endpoint_x = np.where(skeleton & (degree == 1))
    height, width = mask.shape
    grid_y, grid_x = np.ogrid[:height, :width]
    candidates = []
    for label in range(1, count + 1):
        yy, xx = np.where(cores == label)
        peak = float(distance[yy, xx].max())
        central = distance[yy, xx] >= peak * .90
        cx, cy = float(xx[central].mean()), float(yy[central].mean())

        # A terminal disk has short, nearly isotropic radial extent.  Lines have
        # two long tails; arrow corners/wings usually have several skeleton exits.
        rays = []
        limit = max(8., 6. * peak)
        for angle in np.linspace(0, 2*np.pi, 48, endpoint=False):
            radius = 0.
            while radius <= limit:
                x = int(round(cx + radius*np.cos(angle)))
                y = int(round(cy + radius*np.sin(angle)))
                if x < 0 or x >= width or y < 0 or y >= height or not mask[y, x]:
                    break
                radius += .5
            rays.append(radius)
        rays = np.asarray(rays)
        compactness = peak / max(float(np.percentile(rays, 75)), 1e-6)
        long_tail = float(np.mean(rays > 1.8 * peak))

        disk = (grid_x-cx)**2 + (grid_y-cy)**2 <= max(1., .75*peak)**2
        tone = float(darkness[disk].mean()) / ink_peak
        ring_radius = max(3., 1.7*peak)
        squared_radius = (grid_x-cx)**2 + (grid_y-cy)**2
        ring = skeleton & (squared_radius >= ring_radius**2) & (squared_radius < (ring_radius+2.2)**2)
        exits, _ = ndi.label(ring, np.ones((3, 3)))
        exit_count = max(1, int(np.count_nonzero(np.bincount(exits.ravel())[1:])))

        # Measure only the local half-height cap.  Following its connected
        # threshold component along the full stroke makes a small endpoint dot
        # look elongated merely because its one attached line is long.
        cap_radius = max(4., 2.2*peak)
        cap = (squared_radius <= cap_radius**2) & (distance >= max(1.1, .55*peak))
        cap_labels, _ = ndi.label(cap, np.ones((3, 3)))
        region = cap_labels == cap_labels[int(round(cy)), int(round(cx))]
        ry, rx = np.where(region)
        if len(rx) > 2:
            weights = distance[ry, rx] - max(1., .45*peak)
            center = np.average(np.column_stack((rx, ry)), axis=0, weights=weights)
            delta = np.column_stack((rx, ry)) - center
            covariance = (delta*weights[:, None]).T @ delta / weights.sum()
            eigen = np.linalg.eigvalsh(covariance)
            elongation = np.sqrt((eigen[-1]+.5)/(eigen[0]+.5))
        else:
            elongation = 1.
        # The start blob is at a stroke endpoint.  Local maxima at a thick bend or
        # arrow wing can otherwise look round when measured only inside that area.
        endpoint_distance = (float(np.min(np.hypot(endpoint_x-cx, endpoint_y-cy)))
                             if len(endpoint_x) else max(height, width))
        # Skeletonization ends inside a thick terminal disk, not necessarily at
        # its DT center; distance covered by the fitted blob radius is therefore
        # free, while a maximum beyond the endpoint is penalized sharply.
        endpoint_gap = max(0., endpoint_distance - 1.35*peak)
        endpoint_weight = np.exp(-.5*(endpoint_gap/max(1.5, .65*peak))**2)
        score = (peak**2 * compactness**2.2 * tone**1.5 * endpoint_weight *
                 (1-long_tail)**1.5 / (elongation**.70 * exit_count**1.1))
        candidates.append((score, peak, cx, cy))

    if not candidates:
        raise ValueError('No start dot')
    _, peak, cx, cy = max(candidates)
    # Preserve a stable dot signature rather than an arbitrary maximum pixel.
    radius = max(2., 1.25*peak)
    local = ((grid_x-cx)**2 + (grid_y-cy)**2 <= radius**2) & (distance >= .85*peak)
    yy, xx = np.where(local)
    dot = (float(xx.mean()), float(yy.mean()))
    x, y = int(round(dot[0])), int(round(dot[1]))

    # A start disk is terminal even when antialiasing faintly bridges a returning
    # stroke into its opposite side.  In a thin annulus outside the preserved
    # core, retain the strongest substantial attachment and suppress weaker ones.
    # Ordinary closed shapes are unaffected unless a fitted start blob itself has
    # multiple exits, which is precisely the ambiguous topology to resolve.
    radial = np.hypot(grid_x-dot[0], grid_y-dot[1])
    inner = max(2., 1.05*peak)
    annulus = (radial > inner) & (radial <= max(6., 2.8*peak)) & mask
    branches, branch_count = ndi.label(annulus, np.ones((3, 3)))
    if branch_count > 1:
        branch_scores = []
        for branch in range(1, branch_count+1):
            by, bx = np.where(branches == branch)
            if len(bx) >= 2:
                branch_scores.append((float(darkness[by, bx].sum()), branch))
        if len(branch_scores) > 1:
            keep = max(branch_scores)[1]
            mask = mask.copy()
            mask[annulus & (branches != keep)] = False
    distance = ndi.distance_transform_edt(mask)
    labels, _ = ndi.label(mask, np.ones((3, 3)))
    connected = labels == labels[y, x]
    return mask, connected, distance, dot


def pixel_graph(mask):
    sk = skeletonize(mask)
    points = {(int(x), int(y)) for y, x in zip(*np.where(sk))}
    graph = {}
    for x, y in sorted(points):
        neighbors = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = (x + dx, y + dy)
            if q not in points:
                continue
            # Suppress redundant diagonals, not diagonal-only strokes.
            if dx and dy and ((x + dx, y) in points or (x, y + dy) in points):
                continue
            neighbors.append(q)
        graph[x, y] = sorted(neighbors)
    return graph


def compress_graph(graph, start, distance):
    critical = {p for p in graph if len(graph[p]) != 2} | {start}
    # A raster crossing occupies an area, not one pixel. Include its short
    # internal bridges and corner spurs in the same junction, through ink only.
    for p in list(critical):
        if len(graph[p]) < 3 and p != start:
            continue
        radius = max(3.0, 3.0*distance[p[1], p[0]])
        queue = deque([(p, 0.)])
        seen = {p}
        while queue:
            q, length = queue.popleft()
            critical.add(q)
            for r in graph[q]:
                step = float(np.linalg.norm(np.subtract(q,r)))
                if r not in seen and length+step <= radius:
                    seen.add(r)
                    queue.append((r,length+step))
    # Bound by ordinary stroke scale, not the inflated junction radius.
    spur_limit = 8 * float(np.median([distance[y,x] for x,y in graph]))
    # Absorb tiny terminal skeleton spurs inside a sharp ink corner.
    for p in graph:
        if len(graph[p]) < 3:
            continue
        for q in graph[p]:
            chain = [p,q]
            length = float(np.linalg.norm(np.subtract(p,q)))
            while len(graph[q]) == 2 and length <= spur_limit:
                r = next(r for r in graph[q] if r != chain[-2])
                length += float(np.linalg.norm(np.subtract(q,r)))
                chain.append(r)
                q = r
            if len(graph[q]) == 1 and length <= spur_limit:
                critical.update(chain)
    groups = []
    owner = {}
    for p in sorted(critical):
        if p in owner:
            continue
        group = {p}
        todo = [p]
        owner[p] = len(groups)
        while todo:
            a = todo.pop()
            for b in graph[a]:
                if b in critical and b not in owner:
                    owner[b] = len(groups)
                    group.add(b)
                    todo.append(b)
        groups.append(group)
    edges = []
    visited = set()
    for a, group in enumerate(groups):
        for p in sorted(group):
            for q in graph[p]:
                key = frozenset((p, q))
                if q in group or key in visited:
                    continue
                path = [p, q]
                visited.add(key)
                while q not in owner:
                    r = next(r for r in graph[q] if r != path[-2])
                    visited.add(frozenset((q, r)))
                    path.append(r)
                    q = r
                widths = [2 * distance[y, x] for x, y in path]
                # Exclude both junction and endpoint caps when measuring thickness.
                trim = min(max(2, int(np.median(widths) * 1.5)), max(0, (len(path)-3)//2))
                mid = widths[trim:len(widths)-trim] if trim else widths
                edges.append(dict(a=a, b=owner[q], path=path, width=float(np.median(mid)), uses=0, capacity=1))
    return groups, owner, edges


def inside_route(graph, allowed, source, target):
    queue = deque([source])
    parent = {source: None}
    while queue:
        p = queue.popleft()
        if p == target:
            result = []
            while p is not None:
                result.append(p)
                p = parent[p]
            return result[::-1]
        for q in graph[p]:
            if q in allowed and q not in parent:
                parent[q] = p
                queue.append(q)
    raise ValueError('Disconnected junction')


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def tangent(path, width):
    p = np.asarray(path, dtype=float)
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    j = min(len(p)-1, max(1, int(np.searchsorted(length, max(8, width*4)))))
    return unit(p[j]-p[0])


def walk(graph, groups, owner, edges, start):
    adjacency = [[] for _ in groups]
    for i, e in enumerate(edges):
        adjacency[e['a']].append((i, False))
        adjacency[e['b']].append((i, True))
    # Only an unambiguous thick terminal arm of a three-way arrow is repeatable.
    repeat = {}
    for node, incident in enumerate(adjacency):
        if len(incident) != 3:
            continue
        candidates = []
        for i, reverse in incident:
            e = edges[i]
            other = e['a'] if reverse else e['b']
            others = [edges[j]['width'] for j, _ in incident if j != i]
            if len(adjacency[other]) == 1 and other != owner[start] and e['width'] >= 1.25 * max(others) and len(e['path']) > 2*e['width']:
                candidates.append(i)
        if len(candidates) == 1:
            repeat[node] = candidates[0]
            edges[candidates[0]]['capacity'] = 2
    node = owner[start]
    route = [start]
    incoming = None
    history = []
    decisions = []
    for _ in range(2*len(edges)+2):
        options = []
        for i, reverse in adjacency[node]:
            e = edges[i]
            if e['uses'] >= e['capacity']:
                continue
            path = e['path'][::-1] if reverse else e['path']
            direction = tangent(path, e['width'])
            score = float(incoming @ direction) if incoming is not None else -float(direction[1])
            priority = 1 if repeat.get(node) == i and e['uses'] == 0 else 0
            options.append((priority, score, -i, reverse, path, direction))
        if not options:
            break
        _, score, neg_i, reverse, path, direction = max(options, key=lambda o:o[:3])
        i = -neg_i
        e = edges[i]
        if len(options) > 1:
            decisions.append(dict(node=node, edge=i, straight_score=score, arrow_arm=repeat.get(node)==i))
        link = inside_route(graph, groups[node], route[-1], path[0])
        route.extend(link[1:])
        route.extend(path[1:])
        incoming = -tangent(path[::-1], e['width'])
        e['uses'] += 1
        history.append(i)
        node = e['a'] if reverse else e['b']
        if len(adjacency[node]) == 1 and node != owner[start]:
            tips = [p for p in groups[node] if len(graph[p]) == 1]
            if tips:
                tip = max(tips, key=lambda p: np.linalg.norm(np.subtract(p, route[-1])))
                route.extend(inside_route(graph, groups[node], route[-1], tip)[1:])
    return route, history, decisions


def segment_inside(a, b, mask):
    p = np.asarray(a, float)
    q = np.asarray(b, float)
    n = max(2, int(np.linalg.norm(q-p)*4)+1)
    xy = np.rint(p+(q-p)*np.linspace(0, 1, n)[:, None]).astype(int)
    if np.any(xy[:,0] < 0) or np.any(xy[:,1] < 0) or np.any(xy[:,0] >= mask.shape[1]) or np.any(xy[:,1] >= mask.shape[0]):
        return False
    return bool(np.all(mask[xy[:, 1], xy[:, 0]]))


def simplify(route, mask, tolerance=.85):
    """RDP with hard ink support: never replaces a curve by an empty chord."""
    p = np.asarray(route, float)
    keep = {0, len(p)-1}
    stack = [(0, len(p)-1)]
    while stack:
        a, b = stack.pop()
        if b <= a+1:
            continue
        delta = p[b]-p[a]
        t = np.clip((p[a:b+1]-p[a]) @ delta / max(delta@delta, 1e-12), 0, 1)
        error = np.linalg.norm(p[a:b+1]-(p[a]+t[:, None]*delta), axis=1)
        k = a+int(np.argmax(error))
        if error.max() <= tolerance and segment_inside(p[a], p[b], mask):
            continue
        if k in (a, b):
            k = (a+b)//2
        keep.add(k)
        stack.extend(((a, k), (k, b)))
    return p[sorted(keep)].tolist()


def trace(image, tolerance=.85):
    mask, connected, distance, dot = prepare(image)
    graph = pixel_graph(connected)
    if not graph:
        raise ValueError('No skeleton')
    start = min(graph, key=lambda p: (p[0]-dot[0])**2+(p[1]-dot[1])**2)
    groups, owner, edges = compress_graph(graph, start, distance)
    route, history, decisions = walk(graph, groups, owner, edges, start)
    # Pin every entry/exit of a compressed chain so simplification cannot
    # erase a crossing visit or the repeated arrow-arm traversal boundary.
    portals = {p for e in edges for p in (e['path'][0], e['path'][-1])}
    pins = sorted({0,len(route)-1} | {i for i,p in enumerate(route) if p in portals})
    vector = [list(route[0])]
    for a,b in zip(pins,pins[1:]):
        vector.extend(simplify(route[a:b+1], mask, tolerance)[1:])
    visited = set(route)
    # Junction pixels not visited are not missing long strokes.
    missed = sum(max(1, len(e['path'])-1) for e in edges if e['uses']==0)
    total = sum(max(1, len(e['path'])-1) for e in edges)
    dense = np.asarray(route)
    dense_continuous = bool(np.all(np.max(np.abs(np.diff(dense,axis=0)),axis=1)<=1)) if len(dense)>1 else True
    dense_on_ink = bool(np.all(mask[dense[:,1],dense[:,0]]))
    return dict(start_dot=list(dot), start=list(start), end=list(route[-1]),
                path=[list(p) for p in route], vector=vector, vector_segments=len(vector)-1,
                edge_sequence=history, decisions=decisions,
                edges=[{k:v for k,v in e.items() if k!='path'} for e in edges],
                dense_continuous=dense_continuous, dense_on_ink=dense_on_ink,
                skeleton_coverage=len(visited)/len(graph), chain_coverage=1-missed/max(1,total),
                disconnected_ink_pixels=int((mask & ~connected).sum()),
                all_vector_segments_on_ink=all(segment_inside(a,b,mask) for a,b in zip(vector, vector[1:])),
                status='complete' if missed==0 else 'untraced_branches')


def export(image, result, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{stem}.json').write_text(json.dumps(result, indent=2))
    w,h=image.size
    points=' '.join(f'{x:g},{y:g}' for x,y in result['vector'])
    (output/f'{stem}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><polyline points="{points}" fill="none" stroke="#df334a" stroke-width="1" stroke-linejoin="round" stroke-linecap="round"/></svg>')
    im=image.convert('RGB').resize((w*3,h*3))
    draw=ImageDraw.Draw(im)
    p=result['path']
    for i,(a,b) in enumerate(zip(p,p[1:])):
        t=i/max(1,len(p)-2)
        color=(int(235*t),int(145*(1-t)),int(230*(1-t)))
        draw.line((a[0]*3,a[1]*3,b[0]*3,b[1]*3),fill=color,width=2)
    for label,point,color in [('S',p[0],'#008448'),('E',p[-1],'#ce2539')]:
        x,y=point[0]*3,point[1]*3
        draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
        draw.text((x+6,y-12),label,fill=color)
    im.save(output/f'{stem}_overlay.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,default=Path('fresh_output'))
    parser.add_argument('--tolerance',type=float,default=.85)
    args=parser.parse_args()
    files=sorted(args.input.glob('*.png')) if args.input.is_dir() else [args.input]
    summary=[]
    for file in files:
        image=Image.open(file)
        result=trace(image,args.tolerance)
        export(image,result,args.output,file.stem)
        summary.append(dict(file=file.name,**{k:result[k] for k in ('status','chain_coverage','all_vector_segments_on_ink','vector_segments')}))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
