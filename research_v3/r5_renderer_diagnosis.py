"""Read-only dSprites raster replay using the vendored Apache2Sprite vertices.

This produces diagnostic arrays only; it never repairs official RGB or masks.
"""
import os,sys
from v3_common import ROOT,HERE
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib'))
sys.path.insert(0,str(ROOT/'followup/external/svib/data_creation/dsprites'))
from spriteworld.sprite import Sprite
from PIL import Image,ImageDraw
import numpy as np

PALETTE=[[0,255,0],[255,0,255],[0,127,255],[255,127,0]]

def render(objects,sort_objects=None,mask=False):
    basis=objects if sort_objects is None else sort_objects
    order=sorted(range(len(objects)),key=lambda j:(PALETTE.index(basis[j]['color']),basis[j]['2d_coords'][0]))
    canvas=Image.new('RGB',(1280,1280));draw=ImageDraw.Draw(canvas)
    for j in order:
        ob=objects[j];sprite=Sprite(*ob['2d_coords'],shape=ob['shape'],scale=ob['size'],angle=ob['rotation'])
        color=[(0,0,255),(255,0,0)][j] if mask else tuple(ob['color'])
        draw.polygon([tuple(v) for v in sprite.vertices*1280],fill=color)
    return np.flipud(np.asarray(canvas.resize((128,128),Image.Resampling.LANCZOS)))[:,:,::-1].copy()

def compare(actual,expected):
    delta=np.abs(actual.astype(np.int16)-expected.astype(np.int16))
    return {'exact':bool(np.array_equal(actual,expected)),'mae_255':float(delta.mean()),'max_delta_255':int(delta.max()),'different_pixels':int(np.any(delta,axis=-1).sum())}
