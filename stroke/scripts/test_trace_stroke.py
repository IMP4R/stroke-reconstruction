"""Behavior checks, including adversarial gaps and arrow traversal order."""
import unittest
import tempfile
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from trace_stroke import (prepare, pixel_graph, compress_graph, inside_route, unit,
                          tangent, walk, segment_inside, simplify, trace, export)


def drawing(points, width=3, dot=None, extra=()):
    im=Image.new('L',(220,154),255)
    d=ImageDraw.Draw(im)
    d.line(points,fill=150,width=width)
    for p,w in extra:
        d.line(p,fill=150,width=w)
    x,y=dot or points[0]
    d.ellipse((x-5,y-5,x+5,y+5),fill=150)
    return im


class TraceTests(unittest.TestCase):
    def test_dataset_dot_is_not_triangle_corner(self):
        im=Image.open(Path(__file__).resolve().parents[1]/'images/fig_088.png')
        r=trace(im)
        self.assertLess(np.linalg.norm(np.array(r['start'])-[69,25]),5)

    def test_dataset_v_corner_is_not_arrow(self):
        im=Image.open(Path(__file__).resolve().parents[1]/'images/fig_014.png')
        r=trace(im)
        self.assertLess(np.linalg.norm(np.array(r['end'])-[120,113]),4)
        self.assertTrue(all(e['uses']<=1 for e in r['edges']))

    def test_dataset_all_arrow_orders(self):
        root=Path(__file__).resolve().parents[1]
        for n in (25,28,40,58,59,67,77):
            r=trace(Image.open(root/f'images/fig_{n:03}.png'))
            order=r['edge_sequence']
            self.assertEqual(len(order),4,(n,order))
            self.assertEqual(order[1],order[2],(n,order))
            self.assertNotEqual(order[0],order[1])
            self.assertNotEqual(order[2],order[3])
            self.assertEqual(r['status'],'complete')

    def test_prepare_empty(self):
        with self.assertRaises(ValueError): prepare(Image.new('L',(20,20),255))

    def test_trace_bend(self):
        r=trace(drawing([(20,20),(90,120),(180,25)]))
        self.assertLess(np.linalg.norm(np.array(r['start'])-[20,20]),3)
        self.assertLess(np.linalg.norm(np.array(r['end'])-[180,25]),4)
        self.assertEqual(r['status'],'complete')

    def test_pixel_graph_gap(self):
        for gap in (1,2,4,8):
            im=drawing([(20,30),(95,30)],extra=[([(96+gap,30),(190,30)],3)])
            r=trace(im)
            self.assertLess(r['end'][0],96)
            self.assertGreater(r['disconnected_ink_pixels'],100)
            self.assertTrue(r['all_vector_segments_on_ink'])

    def test_walk_plus_stops_without_forced_return(self):
        im=drawing([(20,77),(200,77)],extra=[([(110,20),(110,135)],3)])
        r=trace(im)
        self.assertLess(np.linalg.norm(np.array(r['end'])-[200,77]),4)
        self.assertEqual(r['status'],'untraced_branches')
        self.assertTrue(all(e['uses']<=1 for e in r['edges']))

    def test_walk_crossing_straight(self):
        r=trace(drawing([(20,20),(170,130),(170,20),(20,130)]))
        self.assertEqual(r['status'],'complete')
        p=np.array(r['path'])
        a=np.argmin(np.linalg.norm(p-[170,130],axis=1))
        b=np.argmin(np.linalg.norm(p-[170,20],axis=1))
        self.assertLess(a,b)
        self.assertLess(np.linalg.norm(p[-1]-[20,130]),4)

    def test_walk_thick_arrow_return(self):
        r=trace(drawing([(25,125),(170,30),(170,125)],extra=[([(170,30),(110,30)],6)]))
        counts=Counter(r['edge_sequence'])
        self.assertEqual(sorted(counts.values()),[1,1,2])
        self.assertEqual(r['status'],'complete')
        self.assertLess(np.linalg.norm(np.array(r['end'])-[170,125]),4)

    def test_walk_thin_branch_no_invented_return(self):
        r=trace(drawing([(25,125),(170,30),(170,125)],extra=[([(170,30),(110,30)],3)]))
        self.assertTrue(all(n==1 for n in Counter(r['edge_sequence']).values()))

    def test_simplify_keeps_turnaround(self):
        mask=np.ones((20,20),bool)
        self.assertEqual(simplify([(2,2),(10,2),(2,2)],mask),[[2.,2.],[10.,2.],[2.,2.]])

    def test_segment_inside_forbids_chord(self):
        mask=np.zeros((20,20),bool);mask[2,2:16]=1;mask[2:16,15]=1
        route=[(x,2) for x in range(2,16)]+[(15,y) for y in range(3,16)]
        p=simplify(route,mask,100)
        self.assertGreater(len(p),2)
        self.assertTrue(all(segment_inside(a,b,mask) for a,b in zip(p,p[1:])))

    def test_unit_tangent(self):
        np.testing.assert_allclose(unit([3,4]),[.6,.8])
        np.testing.assert_allclose(tangent([(1,1),(2,1),(3,1)],1),[1,0])

    def test_compress_graph_inside_route(self):
        mask=np.zeros((10,10),bool);mask[5,1:9]=1
        g=pixel_graph(mask)
        groups,owner,edges=compress_graph(g,(1,5),np.ones_like(mask,float))
        self.assertTrue(edges)
        route, history, decisions = walk(g,groups,owner,edges,(1,5))
        self.assertEqual(route[-1],(8,5))
        self.assertEqual(inside_route(g,set(g),(1,5),(8,5))[0],(1,5))

    def test_trace_scaled_inputs(self):
        for scale in (1,2,3):
            im=drawing([(20,20),(90,120),(180,25)])
            im=im.resize((220*scale,154*scale))
            r=trace(im)
            self.assertLess(np.linalg.norm(np.array(r['start'])/scale-[20,20]),4)
            self.assertLess(np.linalg.norm(np.array(r['end'])/scale-[180,25]),4)
            self.assertEqual(r['status'],'complete')

    def test_main_cli(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            drawing([(20,20),(160,100)]).save(p/'input.png')
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('trace_stroke.py')),str(p/'input.png'),'--output',str(p/'out')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((p/'out/summary.json').exists())

    def test_export(self):
        im=drawing([(20,20),(160,100)])
        with tempfile.TemporaryDirectory() as tmp:
            export(im,trace(im),Path(tmp),'test')
            self.assertEqual(len(list(Path(tmp).glob('*'))),3)


if __name__=='__main__': unittest.main()
