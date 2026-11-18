"""BLKH Exp147: reproducible bounded-error RGB rate-distortion benchmark.

Experimental codec, not a production format. All container bytes and quantizer
step-map bytes are included; no learned priors or hidden decoder side info.

Run: python experiment.py [--out PATH] [--fast]
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
import platform
try:
    import resource
except ImportError:
    resource = None
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import brotli
import numpy as np
from PIL import Image, features
from skimage import data

MAGIC = b'B147'
HEADER = struct.Struct('<4sHHBBBHI')
TILE = 16
STEPS = (1, 3, 7, 15, 31)
MODES = (
    ('raw', 'zlib'), ('green_up', 'zlib'), ('green_left', 'zlib'),
    ('raw', 'brotli'), ('green_up', 'brotli'), ('green_left', 'brotli'),
)
DEV = ('camera', 'coins', 'clock', 'grass')
HOLDOUT = ('gravel', 'moon', 'retina', 'cell', 'cat', 'horse')
WEBP_QUALITY = (25, 45, 65, 85, 95, 100)
AVIF_QUALITY = (25, 45, 65, 85, 95, 100)
JXL_DISTANCE = (0., 0.35, 0.7, 1.4, 2.8, 5.)
TARGETS = (35., 40., 45.)


def source(name: str, size: int) -> np.ndarray:
    im = Image.fromarray(getattr(data, name)()).convert('RGB')
    return np.array(im.resize((size, size), resample=Image.Resampling.LANCZOS), dtype=np.uint8)


def step_field(shape, base_step: int, adaptive: bool, original: np.ndarray | None):
    h, w = shape
    ty, tx = (h + TILE - 1) // TILE, (w + TILE - 1) // TILE
    if not adaptive or base_step == 1:
        return np.full((h, w), base_step, np.uint16), b''
    if original is None:
        raise ValueError('adaptive encode needs source')
    # Locked content threshold. 16x16 block activity uses source only at encoder.
    gray = original.astype(np.float32).mean(axis=2)
    ix = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    iy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    bits = np.empty((ty, tx), np.uint8)
    for y in range(ty):
        for x in range(tx):
            patch = (ix + iy)[y*TILE:(y+1)*TILE, x*TILE:(x+1)*TILE]
            bits[y, x] = int(patch.mean() >= 24.0)
    small = max(1, (base_step + 1) // 2)
    if small % 2 == 0:
        small -= 1
    steps = np.full((h, w), base_step, np.uint16)
    for y in range(ty):
        for x in range(tx):
            if bits[y, x]:
                steps[y*TILE:(y+1)*TILE, x*TILE:(x+1)*TILE] = small
    return steps, np.packbits(bits.reshape(-1), bitorder='big').tobytes()


def step_from_map(h, w, base_step, adaptive, packed):
    if not adaptive:
        if packed:
            raise ValueError('unexpected tile map')
        return np.full((h, w), base_step, np.uint16)
    ty, tx = (h+TILE-1)//TILE, (w+TILE-1)//TILE
    count = ty*tx
    if len(packed) != (count+7)//8:
        raise ValueError('invalid tile map')
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder='big')[:count].reshape(ty, tx)
    small = max(1, (base_step+1)//2)
    if small % 2 == 0:
        small -= 1
    steps = np.full((h,w), base_step, np.uint16)
    for y in range(ty):
        for x in range(tx):
            if bits[y,x]:
                steps[y*TILE:(y+1)*TILE,x*TILE:(x+1)*TILE] = small
    return steps


def decorrelate(index: np.ndarray, transform: str) -> np.ndarray:
    if transform == 'raw':
        return index
    b = index.copy()
    b[:,:,0] = (index[:,:,0].astype(np.uint16) - index[:,:,1]) & 255
    b[:,:,2] = (index[:,:,2].astype(np.uint16) - index[:,:,1]) & 255
    if transform == 'green_up':
        d = b.copy()
        d[1:] = (b[1:].astype(np.uint16) - b[:-1]) & 255
    elif transform == 'green_left':
        d = b.copy()
        d[:,1:] = (b[:,1:].astype(np.uint16) - b[:,:-1]) & 255
    else:
        raise ValueError('invalid transform')
    return d


def invert_decorrelation(d: np.ndarray, transform: str) -> np.ndarray:
    if transform == 'raw':
        return d.copy()
    if transform == 'green_up':
        b = np.cumsum(d.astype(np.uint32), axis=0).astype(np.uint8)
    elif transform == 'green_left':
        b = np.cumsum(d.astype(np.uint32), axis=1).astype(np.uint8)
    else:
        raise ValueError('invalid transform')
    a = b.copy()
    a[:,:,0] = (b[:,:,0].astype(np.uint16) + b[:,:,1]) & 255
    a[:,:,2] = (b[:,:,2].astype(np.uint16) + b[:,:,1]) & 255
    return a


def encode(a: np.ndarray, mode: int, step: int, adaptive: bool) -> bytes:
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError('expected uint8 RGB')
    h,w,_ = a.shape
    if not (0 < h <= 65535 and 0 < w <= 65535 and 0 <= mode < len(MODES) and step in STEPS):
        raise ValueError('out of range')
    adaptive = bool(adaptive and step > 1)
    steps, tile_map = step_field((h,w), step, adaptive, a)
    ix = ((a.astype(np.uint16) + steps[:,:,None]//2) // steps[:,:,None]).astype(np.uint8)
    transform, entropy = MODES[mode]
    raw = decorrelate(ix, transform).tobytes()
    payload = zlib.compress(raw, level=6) if entropy == 'zlib' else brotli.compress(raw, quality=5)
    return HEADER.pack(MAGIC,w,h,mode,step,int(adaptive),len(tile_map),len(payload)) + tile_map + payload


def decode(blob: bytes) -> np.ndarray:
    if len(blob) < HEADER.size:
        raise ValueError('truncated header')
    magic,w,h,mode,step,adaptive,maplen,payloadlen = HEADER.unpack_from(blob)
    if (magic!=MAGIC or not w or not h or w*h > 10_000_000 or mode>=len(MODES)
        or step not in STEPS or adaptive not in (0,1) or (step==1 and adaptive)
        or len(blob) != HEADER.size+maplen+payloadlen):
        raise ValueError('invalid file header')
    packed = blob[HEADER.size:HEADER.size+maplen]
    steps = step_from_map(h,w,step,bool(adaptive),packed)
    transform, entropy = MODES[mode]
    payload = blob[HEADER.size+maplen:]
    if entropy == 'zlib':
        dec = zlib.decompressobj()
        raw = dec.decompress(payload, w*h*3+1)
        if not dec.eof or dec.unused_data or dec.unconsumed_tail:
            raise ValueError('invalid compressed payload')
    else:
        raw = brotli.decompress(payload)
    if len(raw)!=w*h*3:
        raise ValueError('incorrect decoded length')
    ix = invert_decorrelation(np.frombuffer(raw,dtype=np.uint8).reshape(h,w,3), transform)
    return np.minimum(ix.astype(np.uint16)*steps[:,:,None], 255).astype(np.uint8)


def metrics(a, b):
    delta = a.astype(np.int16)-b.astype(np.int16)
    mse = float(np.mean(delta.astype(np.float64)**2))
    return {'psnr_db': 99. if mse==0 else round(10*math.log10(255**2/mse), 5),
            'mse':round(mse,6), 'max_abs_error':int(np.abs(delta).max()),
            'exact':bool(np.array_equal(a,b))}


def pil_codec(a, name, quality):
    buf=io.BytesIO()
    kwargs=({'quality':int(quality),'method':6} if name=='WEBP' else {'quality':int(quality),'speed':8})
    start=time.perf_counter()
    Image.fromarray(a).save(buf,format=name,**kwargs)
    blob=buf.getvalue()
    enc=time.perf_counter()-start
    start=time.perf_counter()
    with Image.open(io.BytesIO(blob)) as out:
        rec=np.asarray(out.convert('RGB'),dtype=np.uint8)
    dec=time.perf_counter()-start
    if rec.shape!=a.shape:
        raise AssertionError(f'{name} wrong dimensions')
    return blob,rec,enc,dec


def jxl_codec(a, distance, ffmpeg):
    h,w,_=a.shape
    enc_cmd=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo',
             '-pixel_format','rgb24','-video_size',f'{w}x{h}','-i','pipe:0',
             '-frames:v','1','-c:v','libjxl','-distance',str(distance),'-effort','3',
             '-f','image2pipe','pipe:1']
    dec_cmd=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-f','image2pipe',
             '-c:v','libjxl','-i','pipe:0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    t=time.perf_counter()
    e=subprocess.run(enc_cmd,input=a.tobytes(),capture_output=True,timeout=60)
    enc=time.perf_counter()-t
    if e.returncode or not e.stdout:
        raise RuntimeError('JPEG XL encode failed: '+e.stderr.decode(errors='replace')[:500])
    t=time.perf_counter()
    d=subprocess.run(dec_cmd,input=e.stdout,capture_output=True,timeout=60)
    dec=time.perf_counter()-t
    if d.returncode or len(d.stdout)!=w*h*3:
        raise RuntimeError('JPEG XL decode failed: '+d.stderr.decode(errors='replace')[:500])
    return e.stdout,np.frombuffer(d.stdout,dtype=np.uint8).copy().reshape(h,w,3),enc,dec


def rd_at(points, target):
    """Interpolate log(bytes) by achieved PSNR; no extrapolation, Pareto envelope.
    Min file size among all runs with quality >= target defines the monotone frontier.
    """
    s=sorted((float(p['psnr_db']),int(p['bytes'])) for p in points)
    values=[]
    for q,b in s:
        if not values or q != values[-1][0]:
            values.append([q,b])
        else:
            values[-1][1] = min(b,values[-1][1])
    for i in range(len(values)-2,-1,-1):
        values[i][1]=min(values[i][1],values[i+1][1])
    # Discrete optimum if an existing setting clears the target, and interpolated
    # estimate is reported separately; neither is an actual attained byte count.
    qualifying=[p['bytes'] for p in points if p['psnr_db']>=target]
    at_or_above=min(qualifying) if qualifying else None
    interp=None
    for (q1,b1),(q2,b2) in zip(values[:-1],values[1:]):
        if q1<=target<=q2 and q2>q1 and b1>0 and b2>0:
            alpha=(target-q1)/(q2-q1)
            interp=round(math.exp(math.log(b1)*(1-alpha)+math.log(b2)*alpha),1)
            break
    return {'measured_min_bytes_above_target':at_or_above,
            'interpolated_bytes':interp,'in_range':interp is not None}


def self_tests():
    rng=np.random.default_rng(147)
    sources=[rng.integers(0,256,(35,39,3),dtype=np.uint8),
             np.zeros((33,40,3),dtype=np.uint8),
             np.full((17,19,3),255,dtype=np.uint8)]
    checks=0
    for a in sources:
        for mode in range(len(MODES)):
            for step in STEPS:
                for adaptive in (False,True):
                    if step==1 and adaptive:
                        continue
                    blob=encode(a,mode,step,adaptive)
                    rec=decode(blob)
                    err=np.abs(a.astype(np.int16)-rec.astype(np.int16))
                    assert err.max()<=step//2,(mode,step,adaptive,err.max())
                    if step==1:assert np.array_equal(a,rec)
                    checks+=1
    sample=encode(sources[0],0,3,False)
    invalid=[b'',b'B147',b'XXXX'+sample[4:],sample[:-1],sample+b'X',
             sample[:4]+b'\0\0'+sample[6:]]
    for v in invalid:
        try:
            decode(v)
        except (ValueError,zlib.error,brotli.error):
            pass
        else:
            raise AssertionError('malformed input accepted')
    return {'valid_roundtrips':checks,'malformed_rejected':len(invalid)}


def run(out: Path,fast=False):
    tests=self_tests()
    names=DEV+HOLDOUT if not fast else (DEV[:2]+HOLDOUT[:2])
    steps=STEPS if not fast else (1,7,15)
    ffmpeg=shutil.which('ffmpeg')
    env={'python':platform.python_version(),'numpy':np.__version__,
         'pillow':Image.__version__,'libwebp':features.version('webp'),
         'brotli':getattr(brotli,'__version__','unknown'),
         'ffmpeg':ffmpeg,'skimage_version':__import__('skimage').__version__,
         'system':platform.platform(),'size':192,'resampler':'Pillow.LANCZOS',
         'ffmpeg_libjxl_available':bool(ffmpeg),'AVIF_encoder':True,
         'max_process_rss_kib':None,
         'data_source':'scikit-image bundled images; no external download'}
    results=[]
    for name in names:
        a=source(name,192)
        split='development' if name in DEV else 'holdout'
        row={'image':name,'split':split,'sha256_rgb':hashlib.sha256(a.tobytes()).hexdigest(),
             'dimensions':[192,192],'BLKH':[],'WEBP':[],'AVIF':[],'JPEGXL':[]}
        for step in steps:
            for adaptive in (False,True):
                if step==1 and adaptive:continue
                for mode in range(len(MODES)):
                    start=time.perf_counter()
                    blob=encode(a,mode,step,adaptive)
                    et=time.perf_counter()-start
                    start=time.perf_counter()
                    rec=decode(blob)
                    dt=time.perf_counter()-start
                    m=metrics(a,rec)
                    assert m['max_abs_error']<=step//2
                    if step==1:assert m['exact']
                    row['BLKH'].append({'mode':mode,'transform':MODES[mode][0],
                                        'entropy':MODES[mode][1],
                                        'step':step,'adaptive':adaptive,'bytes':len(blob),
                                        'encode_s':round(et,6),'decode_s':round(dt,6),**m})
        for q in WEBP_QUALITY if not fast else (45,85,100):
            blob,rec,et,dt=pil_codec(a,'WEBP',q)
            row['WEBP'].append({'quality':q,'bytes':len(blob),
                                 'encode_s':round(et,6),'decode_s':round(dt,6),**metrics(a,rec)})
        for q in AVIF_QUALITY if not fast else (45,85,100):
            blob,rec,et,dt=pil_codec(a,'AVIF',q)
            row['AVIF'].append({'quality':q,'bytes':len(blob),
                                 'encode_s':round(et,6),'decode_s':round(dt,6),**metrics(a,rec)})
        if ffmpeg:
            for d in JXL_DISTANCE if not fast else (0.,1.,4.):
                blob,rec,et,dt=jxl_codec(a,d,ffmpeg)
                m=metrics(a,rec)
                if d==0:assert m['exact'],f'JXL lossless not exact {name}'
                row['JPEGXL'].append({'distance':d,'bytes':len(blob),
                                      'encode_s':round(et,6),'decode_s':round(dt,6),**m})
        results.append(row)
        print(f"{split} {name}: {len(row['BLKH'])} BLKH, {len(row['WEBP'])} WebP, {len(row['AVIF'])} AVIF, {len(row['JPEGXL'])} JXL",flush=True)
    dev=[r for r in results if r['split']=='development']
    hold=[r for r in results if r['split']=='holdout']
    # Lock model on dev at 40dB, using only interpolated BLKH rate estimates.
    choices=[]
    for mode in range(len(MODES)):
        for adapt in (False,True):
            rates=[]
            for row in dev:
                p=[x for x in row['BLKH'] if x['mode']==mode and x['adaptive']==adapt]
                v=rd_at(p,40.)['interpolated_bytes']
                if v is not None: rates.append(v)
            if len(rates)==len(dev):
                choices.append((sum(math.log(v) for v in rates),mode,adapt))
    if not choices:raise AssertionError('no development configuration supports 40dB')
    _,chosen_mode,chosen_adaptive=min(choices)
    per_image=[]
    for row in hold:
        p=[x for x in row['BLKH'] if x['mode']==chosen_mode and x['adaptive']==chosen_adaptive]
        rd={str(t):{codec:rd_at(p if codec=='BLKH' else row[codec],t)
                    for codec in ('BLKH','WEBP','AVIF','JPEGXL')} for t in TARGETS}
        lossless_blkh=min(x['bytes'] for x in p if x['step']==1)
        webp_lossless=io.BytesIO()
        Image.fromarray(source(row['image'],192)).save(webp_lossless,format='WEBP',lossless=True,quality=100,method=6)
        decoded_webp=np.asarray(Image.open(io.BytesIO(webp_lossless.getvalue())).convert('RGB'))
        assert np.array_equal(decoded_webp,source(row['image'],192))
        lossless_jxl=next((x['bytes'] for x in row['JPEGXL'] if x['distance']==0),None)
        per_image.append({'image':row['image'], 'lossless_bytes':{'BLKH':lossless_blkh,
                          'WEBP':len(webp_lossless.getvalue()),'JPEGXL':lossless_jxl},
                          'matched_quality':rd})
    lossless_sums={k:sum(x['lossless_bytes'][k] for x in per_image if x['lossless_bytes'][k] is not None)
                   for k in ('BLKH','WEBP','JPEGXL')}
    comparisons={}
    for target in TARGETS:
        comparisons[str(target)]={}
        for baseline in ('WEBP','AVIF','JPEGXL'):
            pairs=[]
            for item in per_image:
                a=item['matched_quality'][str(target)]['BLKH']['interpolated_bytes']
                b=item['matched_quality'][str(target)][baseline]['interpolated_bytes']
                if a is not None and b is not None:pairs.append((item['image'],a,b))
            comparisons[str(target)][baseline]={'coverage':len(pairs),'total_holdout':len(hold),
                'blkh_interpolated_bytes':round(sum(p[1] for p in pairs),2),
                'baseline_interpolated_bytes':round(sum(p[2] for p in pairs),2),
                'relative_delta_percent':round((sum(p[1] for p in pairs)/sum(p[2] for p in pairs)-1)*100,2) if pairs else None,
                'images':[p[0] for p in pairs]}
    env['max_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
    obj={'experiment':'exp147','dataset':{'development':list(DEV),'holdout':list(HOLDOUT),
         'evaluated':list(names),'train_test_identity_overlap':False,
         'prior_reuse_possible':True,'third_party_independent':False},
         'settings':{'steps':steps,'tile':TILE,'modes':MODES,
                     'development_selection_target_psnr_db':40.,'rd_targets_db':TARGETS,
                     'AVIF_quality':AVIF_QUALITY,'WEBP_quality':WEBP_QUALITY,
                     'JXL_distance':JXL_DISTANCE,'random_seed':147,
                     'lossless_definition':'exact RGB pixel equality',
                     'interpolation':'log(bytes) vs PSNR; no extrapolation; not measured operating point'},
         'environment':env,'tests':tests,'selected_on_development':{'mode':chosen_mode,
                     'name':'/'.join(MODES[chosen_mode]),'adaptive':chosen_adaptive},
         'holdout_lossless_total_bytes':lossless_sums,
         'holdout_matched_quality':comparisons,
         'holdout_detailed':per_image,'results':results,
         'limitations':['Bundled scikit-image images and resized 192x192; not external unseen corpus',
            'Independent image identities between dev/test but some may appear in historical BLKH experiments',
            'AVIF and WebP lossy encode settings are not quality-equivalent by number; PSNR matching is used',
            'RD interpolation is an estimate, not a bitstream actually encoded at target PSNR',
            'Process RSS includes Python runtime and all codec dependencies',
            'Local run only unless exact GitHub Actions run explicitly confirmed']}
    out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    lines=[f'# BLKH Exp147 — Bounded-error quantization vs modern codecs',
           '', '## Experimental hypotheses',
           'Uniform 8-bit RGB scalar quantization (step 1/3/7/15/31) vs 16x16 adaptive edge-aware step selection; six reversible predictors+entropy backends.',
           'Full on-disk B147 header, mode, quantizer, side-map, and compressed payload counted. Reconstruction is bounded to <= floor(step/2) absolute RGB error; step=1 exact.',
           '', '## Separation',
           f'Development: {", ".join(DEV)}. Holdout: {", ".join(HOLDOUT)}. All original skimage data resized to 192x192 LANCZOS RGB.',
           'Mode and quantizer selected only from development images at 40 dB (interpolated rate); holdout untouched for selection.',
           '', '## Selection and outcome',
           f'Selected: `{MODES[chosen_mode]}` adaptive={chosen_adaptive}. Self-tests: {tests}.',
           f'Holdout lossless full-file totals: {lossless_sums}.',
           '', '## Rate-distortion (interpolated bytes at equal PSNR)',
           '| Target PSNR | Codec | Coverage | BLKH bytes | Baseline bytes | BLKH excess |',
           '|---:|---|---:|---:|---:|---:|']
    for t, comps in comparisons.items():
        for code,v in comps.items():
            lines.append(f"| {t} dB | {code} | {v['coverage']}/{v['total_holdout']} | {v['blkh_interpolated_bytes']} | {v['baseline_interpolated_bytes']} | {v['relative_delta_percent']}% |")
    lines += ['', '## Integrity',
              f"{tests['valid_roundtrips']} randomized/edge self-test roundtrips plus {sum(len(r['BLKH']) for r in results)} full-image encode/decode validations; {tests['malformed_rejected']} malformed files rejected.",
              'All candidate RGB images decoded and max-error checked. WebP/AVIF/JXL decoded and evaluated on actual RGB outputs; JXL distance=0 pixel-exact.',
              '', '## Known limits', *('- '+x for x in obj['limitations']),
              '', '## Reproduce',
              '`python research/exp147_rd/experiment.py --out research/exp147_rd/outputs`',
              'Requires Python packages numpy, pillow[AVIF support], scikit-image, brotli and ffmpeg compiled with libjxl.',
              f'Environment: {env}', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'selected':obj['selected_on_development'],
          'lossless_total':lossless_sums,'matched_quality':comparisons,'self_tests':tests},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).parent)
    parser.add_argument('--fast',action='store_true')
    args=parser.parse_args()
    run(args.out,args.fast)