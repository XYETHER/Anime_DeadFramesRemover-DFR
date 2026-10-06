# SPDX-License-Identifier: MIT
# Copyright (c) 2026 xyether
#@title 2. DFR — Remove dead frames
"""DFR decision core: depth is attention, original pixels are evidence."""
import cv2
import numpy as np

def foreground(depth, focus=50):
    """Relative inverse depth: larger=nearer. None signals unusable depth."""
    if depth is None or not np.isfinite(depth).all(): return None
    lo,hi=np.percentile(depth,[2,98])
    if hi-lo < max(1e-6,abs(float(np.median(depth)))*.005): return None
    d=np.uint8(np.clip((depth-lo)/(hi-lo),0,1)*255)
    threshold,_=cv2.threshold(d,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    threshold=float(np.clip(threshold+(focus-50)*1.4,25,225))
    mask=np.uint8(d>threshold)
    scale=max(1.,max(depth.shape)/640)
    close_size=int(round(7*scale))|1;erode_size=int(round(3*scale))|1
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((close_size,close_size),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    clean=np.zeros_like(mask)
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA]>=max(9,mask.size*.00005): clean[labels==i]=1
    clean=cv2.erode(clean,np.ones((erode_size,erode_size),np.uint8))
    if clean.mean()<.025 or clean.mean()>.96:return None
    return clean.astype(bool)

def prepare(frame):
    return cv2.GaussianBlur(frame.astype(np.float32),(3,3),.65)

def difference(a,b,mask,strength=50):
    """Higher strength tolerates more change and removes more frames."""
    if mask is None:mask=np.ones(a.shape[:2],bool)
    diff=np.max(np.abs(a-b),axis=2)
    # A local patch gate protects small eye/mouth motion from global dilution.
    pixel=2.0+strength*.08
    active=(diff>pixel)&mask
    area=max(1,int(mask.sum()))
    fraction=float(active.sum()/area)
    # Keep the spatial meaning of sensitivity consistent at 640px and 1080p.
    scale=max(1.,max(mask.shape)/640);large_size=round(20*scale);tiny_size=round(8*scale)
    mass=cv2.boxFilter(mask.astype(np.float32),-1,(large_size,large_size),normalize=False)
    changed=cv2.boxFilter(active.astype(np.float32),-1,(large_size,large_size),normalize=False)
    valid=mass>large_size*large_size*.4
    local=float(np.max(changed[valid]/mass[valid])) if valid.any() else 0.
    global_limit=.004+strength*.00020
    local_limit=.08+strength*.0024
    # Separate high-contrast small-detail gate: a brief blink must not be
    # diluted by a 20x20 window or wait to accumulate over several frames.
    tiny=((diff>max(12.,pixel*2))&mask).astype(np.float32)
    tiny_mass=cv2.boxFilter(mask.astype(np.float32),-1,(tiny_size,tiny_size),normalize=False)
    tiny_count=cv2.boxFilter(tiny,-1,(tiny_size,tiny_size),normalize=False)
    valid_tiny=tiny_mass>=tiny_size*tiny_size*.5
    tiny_peak=float(np.max(tiny_count[valid_tiny]/tiny_mass[valid_tiny])) if valid_tiny.any() else 0.
    score=max(fraction/global_limit,local/local_limit,tiny_peak/(.10+strength*.001))
    return float(score),fraction,local

def is_cut(a,b):
    """Only strong, widespread discontinuities count; motion handles others."""
    x=cv2.resize(a,(96,54));y=cv2.resize(b,(96,54))
    d=np.mean(np.abs(x-y),axis=2)
    return bool(np.mean(d)>48 and np.mean(d>25)>.70)

class Selector:
    def __init__(self,strength=50):
        self.strength=strength;self.anchor=None;self.prev=None;self.mask=None;self.index=-1
    def step(self,frame,mask,index):
        f=prepare(frame);cut=self.prev is not None and is_cut(self.prev,f)
        reason='first' if self.anchor is None else ('cut' if cut else '')
        unreliable=mask is None or self.mask is None
        region=None if unreliable else mask|self.mask
        score,frac,local=(0.,0.,0.) if self.anchor is None else difference(self.anchor,f,region,self.strength)
        # Flat/unreliable depth falls back to stricter full-frame comparison.
        if unreliable and self.anchor is not None:
            score,frac,local=difference(self.anchor,f,None,min(self.strength,25))
        keep=bool(reason or score>=1)
        anchor_index=self.index
        if keep:self.anchor=f;self.mask=mask;self.index=index
        self.prev=f
        return {'frame':index,'keep':keep,'reference':anchor_index,'reason':reason or ('motion' if keep else 'held'),'score':score,'changed_fraction':frac,'local_change':local,'mask_area':float(mask.mean()) if mask is not None else 1.,'fallback':unreliable}

def self_test():
    rng=np.random.default_rng(7);a=np.zeros((180,320,3),np.uint8)+50
    a[40:160,100:230]=150;mask=np.zeros((180,320),bool);mask[45:155,105:225]=1
    z=Selector();assert z.step(a,mask,0)['keep'];assert not z.step(a,mask,1)['keep']
    b=a.copy();b[:,:70]=230;assert not z.step(b,mask,2)['keep'],'background-only'
    b=a.copy();b[80:84,150:168]=10;assert z.step(b,mask,3)['keep'],'eye motion'
    z=Selector();z.step(a,mask,0);noise=np.clip(a.astype(float)+rng.normal(0,.6,a.shape),0,255).astype('uint8');assert not z.step(noise,mask,1)['keep'],'codec-like noise'
    z=Selector();z.step(a,mask,0)
    found=False
    for i in range(1,21):
        b=a.copy();b[70:100,140:180]+=i
        if z.step(b,mask,i)['keep']:found=True;break
    assert found,'slow cumulative change lost'
    for hh,ww in [(1,8),(2,12),(3,12)]:
        z=Selector();z.step(a,mask,0);blink=a.copy();blink[80:80+hh,150:150+ww]=0
        assert z.step(blink,mask,1)['keep'],('tiny blink',hh,ww)
        assert z.step(a,mask,2)['keep'],'return from blink'
    assert foreground(np.ones((180,320))) is None
    z=Selector();z.step(a,mask,0);assert z.step(255-a,mask,1)['keep'],'cut lost'
    for strength in [0,25,50,75,100]:
        score,*_=difference(prepare(a),prepare(b),mask,strength)
        if strength:assert score<=last+1e-6
        last=score
    print('PASS: duplicate, independent background, eye, noise, cumulative change, flat depth, cut, sensitivity')


import pathlib,time,json,csv,subprocess,os

def analyse_video(path,depth_fn,strength=50,focus=50,output_dir='/content/DFR/results',save_cache=False,analysis_width=640,frame_step=1):
    path=pathlib.Path(path);out=pathlib.Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if frame_step not in (1,2,3):raise ValueError("Invalid frame step")
    timing=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))
    timestamps=[float(f['best_effort_timestamp_time']) for f in timing.get('frames',[]) if 'best_effort_timestamp_time' in f]
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError('Cannot decode '+str(path))
    w,h=int(cap.get(3)),int(cap.get(4));total=int(cap.get(7))
    scale=min(1.,analysis_width/max(w,h));aw,ah=max(2,round(w*scale)),max(2,round(h*scale))
    sel=Selector(strength);kept_count=0;rows=[];shots=[];cached=[];last_frame=None;mask=None;depth=None;depth_calls=0;t=time.perf_counter()
    stage={k:0. for k in ['decode_s','resize_s','depth_s','foreground_s','decision_s']}
    try:
        while True:
            stamp=time.perf_counter();i=len(rows);ok=cap.grab()
            if not ok:break
            if i%frame_step:
                stage['decode_s']+=time.perf_counter()-stamp
                row=dict(rows[-1]);row.update(frame=i,keep=False,reason='pre_skip',output_frame='',score=0.,changed_fraction=0.,local_change=0.,mask_area=0.,fallback=False,source_time_s=timestamps[i] if i<len(timestamps) else float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
                rows.append(row);continue
            ok,f=cap.retrieve()
            if not ok:raise RuntimeError('Frame retrieval failed')
            stage['decode_s']+=time.perf_counter()-stamp
            stamp=time.perf_counter();small=cv2.resize(f,(aw,ah),interpolation=cv2.INTER_AREA);i=len(rows);stage['resize_s']+=time.perf_counter()-stamp
            # Exact low-resolution duplicates need no repeated network inference.
            if last_frame is None or not np.array_equal(small,last_frame):
                stamp=time.perf_counter();depth=depth_fn(small);stage['depth_s']+=time.perf_counter()-stamp
                stamp=time.perf_counter();mask=foreground(depth,focus);stage['foreground_s']+=time.perf_counter()-stamp;depth_calls+=1
            stamp=time.perf_counter();row=sel.step(small,mask,i);stage['decision_s']+=time.perf_counter()-stamp
            row['source_time_s']=timestamps[i] if i<len(timestamps) else float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            row['output_frame']=kept_count if row['keep'] else ''
            kept_count+=int(row['keep'])
            rows.append(row)
            if save_cache:cached.append((small.copy(),depth.copy()))
            if i in set([0,1,max(1,total//3),max(1,total*2//3),max(1,total-2)]):
                shots.append((i,small.copy(),None if mask is None else mask.copy()))
            last_frame=small
            if i%100==0: print(path.name,i,'/',total,flush=True)
    finally:cap.release()
    if not rows:raise ValueError('No frames decoded')
    if timestamps and len(rows)!=len(timestamps):raise RuntimeError('Decoder/frame timestamp count mismatch; refusing partial output')
    # Decoder frame count is the ground truth; metadata can be wrong on VFR.
    analysis_seconds=time.perf_counter()-t
    csvpath=out/(path.stem+'_DFR_decisions.csv')
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(aw*2,(ah+24)*len(shots)),(20,20,20));draw=ImageDraw.Draw(sheet)
    for j,(i,im,m) in enumerate(shots):
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);vis=rgb.copy()
        if m is not None:vis[~m]=(vis[~m]*.16).astype('uint8')
        sheet.paste(Image.fromarray(rgb),(0,j*(ah+24)+24));sheet.paste(Image.fromarray(vis),(aw,j*(ah+24)+24))
        draw.text((4,j*(ah+24)+5),f'Frame {i} | original / attention region (NOT output) | '+rows[i]['reason'],fill='white')
    sheet.save(out/(path.stem+'_DFR_attention.jpg'))
    summary={'name':path.name,'input_frames':len(rows),'kept_frames':sum(r['keep'] for r in rows),'analysis_seconds':analysis_seconds,'analysis_fps':len(rows)/analysis_seconds,'depth_calls':depth_calls,'fallback_frames':sum(r['fallback'] for r in rows),'vfr_detected':bool(len(timestamps)>2 and np.ptp(np.diff(timestamps))>.001),'cuts':[r['frame'] for r in rows if r['reason']=='cut'],'strength':strength,'focus':focus,'width':w,'height':h}
    summary['stage_seconds']=stage;summary['analysis_size']=[aw,ah]
    summary['frame_step']=frame_step;summary['analysed_frames']=sum(r['reason']!='pre_skip' for r in rows)
    (out/(path.stem+'_DFR_summary.json')).write_text(json.dumps(summary,indent=2))
    if save_cache:
        np.savez_compressed(out/(path.stem+'_cache.npz'),frames=np.stack([v[0] for v in cached]),depths=np.stack([v[1] for v in cached]).astype(np.float16))
    print(json.dumps(summary),flush=True)
    return rows,summary

def encode_video(path,rows,output_dir='/content/DFR/results',fps_override=0,encoder='h264_nvenc'):
    from fractions import Fraction
    path=pathlib.Path(path);out=pathlib.Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(path)]))['streams'][0]
    rate=str(fps_override) if fps_override else probe.get('avg_frame_rate','0/0')
    if rate in ['0/0','0']:rate=probe.get('r_frame_rate','24/1')
    fps=float(Fraction(rate));assert 0<fps<=240,rate
    cap=cv2.VideoCapture(str(path));w,h=int(cap.get(3)),int(cap.get(4))
    final=out/(path.stem+'_DFR.mp4');temp=out/(path.stem+'_DFR.partial.mp4')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-framerate',rate,'-i','pipe:0','-an']
    if encoder not in ['h264_nvenc','hevc_nvenc']:
        raise ValueError('DFR supports NVIDIA hardware encoding only: H.264 or H.265')
    cmd += ['-c:v',encoder,'-preset','p7','-tune','hq','-rc','vbr','-cq','22','-b:v','0',
            '-multipass','fullres','-rc-lookahead','32','-spatial-aq','1','-temporal-aq','1',
            '-aq-strength','8','-bf','3','-b_ref_mode','middle','-g','240']
    if encoder=='hevc_nvenc':cmd += ['-tag:v','hvc1']
    cmd += ['-pix_fmt','yuv420p' if w%2==0 and h%2==0 else 'yuv444p','-movflags','+faststart']
    for flag,key in [('-color_primaries','color_primaries'),('-color_trc','color_transfer'),('-colorspace','color_space')]:
        if probe.get(key) not in [None,'unknown','reserved','unspecified']:cmd += [flag,probe[key]]
    # Explicit RGB-to-YUV matrix; merely copying colour tags does not
    # configure conversion and can miscolour HD output.
    matrix='bt709' if probe.get('color_space','bt709') in ['bt709','unknown'] else 'bt601'
    if probe.get('color_space') in [None,'unknown','unspecified']:
        cmd += ['-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709']
    cmd += ['-vf','scale=in_range=full:out_range=tv:out_color_matrix='+matrix,'-color_range','tv',str(temp)]
    start=time.perf_counter();n=0;i=0;log=out/(path.stem+'_encode.log')
    try:
        with log.open('wb') as stderr:
            enc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=stderr)
            try:
                while True:
                    ok=cap.grab()
                    if not ok:break
                    if i>=len(rows):raise RuntimeError('Input changed between analysis and encoding')
                    if rows[i]['keep']:
                        ok,f=cap.retrieve()
                        if not ok:raise RuntimeError('Frame retrieval failed')
                        enc.stdin.write(f.tobytes());n+=1
                    i+=1
                enc.stdin.close();ret=enc.wait(timeout=180)
                if ret:raise RuntimeError('Hardware encoder failed. No CPU fallback. '+log.read_text())
            except BaseException:
                enc.kill();enc.wait();raise
        if i!=len(rows):raise RuntimeError('Decode ended before all analyzed frames')
        check=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(temp)]))
        v=next(s for s in check['streams'] if s['codec_type']=='video')
        assert int(v['nb_read_frames'])==n and int(v['width'])==w and int(v['height'])==h,check
        assert Fraction(v['avg_frame_rate'])==Fraction(rate),v
        assert v['codec_name']==('hevc' if encoder=='hevc_nvenc' else 'h264'),v
        assert all(s['codec_type']!='audio' for s in check['streams'])
        temp.replace(final)
        result={'output':str(final),'output_frames':n,'output_fps':rate,'output_duration':n/fps,'encode_seconds':time.perf_counter()-start,'encoder':encoder,'verified':True}
        print(json.dumps(result),flush=True);return result
    finally:
        cap.release()
        if temp.exists():temp.unlink()
