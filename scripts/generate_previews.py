# SPDX-License-Identifier: MIT
# Copyright (c) 2026 xyether
# Run inside Colab; this script reads the notebook and uses its controls.
import json, os, pathlib, subprocess, sys, time
from IPython.core.interactiveshell import InteractiveShell
import argparse, shutil
parser=argparse.ArgumentParser(description="Reproduce xyether's DFR demo on a Colab GPU")
parser.add_argument('input',type=pathlib.Path)
parser.add_argument('--output-dir',type=pathlib.Path,default=pathlib.Path('/content/DFR_demo'))
args=parser.parse_args()
ROOT=args.output_dir.resolve();ROOT.mkdir(parents=True,exist_ok=True)
if args.input.resolve()!=ROOT/'input.mp4':shutil.copy2(args.input,ROOT/'input.mp4')
OUT=ROOT/'previews'; OUT.mkdir(exist_ok=True)
nb=json.loads((pathlib.Path(__file__).resolve().parents[1]/'DFR_Colab.ipynb').read_text(encoding='utf-8'))
shell=InteractiveShell.instance()
ns=shell.user_ns
def cell(code):
    result=shell.run_cell(code)
    if result.error_before_exec: raise result.error_before_exec
    if result.error_in_exec: raise result.error_in_exec

print('PHASE DFR setup',flush=True)
cell(''.join(nb['cells'][1]['source']))
cell(''.join(nb['cells'][2]['source']).split('#@markdown **Input:**')[0])
cell("self_test()\nDFR_results=dfr_run("+repr(str(ROOT/'input.mp4'))+",codec='H.264 (NVIDIA)',auto_download=False,processing_type='Type 1')")
report=ns['DFR_results'][0]
import shutil
shutil.copy2(report['output'],OUT/'dfr.mp4')
shutil.copy2(pathlib.Path(report['output']).parent/'input_DFR_decisions.csv', OUT/'decisions.csv')
shutil.copy2(pathlib.Path(report['output']).parent/'input_DFR_attention.jpg', OUT/'attention.jpg')
print('PHASE depth preview',flush=True)
import cv2, numpy as np, torch
model=ns['DFR_MODEL']
cap=cv2.VideoCapture(str(ROOT/'input.mp4')); w=int(cap.get(3)); h=int(cap.get(4)); fps=cap.get(5)
cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','640x360','-framerate',str(fps),'-i','pipe:0','-an','-c:v','h264_nvenc','-preset','p7','-tune','hq','-rc','vbr','-cq','19','-b:v','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'depth.mp4')]
enc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
i=0; last=None; depth=None
with torch.inference_mode():
    while True:
        ok,frame=cap.read()
        if not ok:break
        small=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
        if last is None or not np.array_equal(small,last):
            with torch.autocast('cuda',dtype=torch.float16):
                tensor,_=model.image2tensor(small,252); depth=model(tensor)[0]
            depth=cv2.resize(depth.float().cpu().numpy(),(640,360))
        lo,hi=np.percentile(depth,[2,98]); grey=np.uint8(np.clip((depth-lo)/max(hi-lo,1e-8),0,1)*255)
        color=cv2.applyColorMap(grey,cv2.COLORMAP_INFERNO)
        enc.stdin.write(color.tobytes())
        if i in (0,64,128,192):
            cv2.imwrite(str(OUT/f'frame-{i:03d}.jpg'),small)
            cv2.imwrite(str(OUT/f'depth-{i:03d}.jpg'),color)
            mask=ns['foreground'](depth,50)
            att=small.copy()
            if mask is not None:att[~mask]=(att[~mask]*.16).astype('uint8')
            cv2.imwrite(str(OUT/f'attention-{i:03d}.jpg'),att)
        last=small;i+=1
cap.release();enc.stdin.close()
if enc.wait()!=0:raise RuntimeError('Depth preview encoder failed')
report['depth_preview_frames']=i
print('PHASE RIFE setup',flush=True)
setup=''.join(nb['cells'][4]['source']).replace('Auto_Reconnect = True','Auto_Reconnect = False')
cell(setup)
print('PHASE repeat RIFE setup validation',flush=True)
cell(setup)
cell(''.join(nb['cells'][5]['source']).split('# ------------------- Processing params')[0])
ns['os'].environ['RIFE_ENCODER']='h264_nvenc'
for name,source in [('input-interpolated',ROOT/'input.mp4'),('dfr-interpolated',OUT/'dfr.mp4')]:
    print('PHASE '+name,flush=True)
    t=time.perf_counter()
    result=ns['interpolate_video'](str(source),'RIFE_v4.26',8,60)
    shutil.copy2(result,OUT/(name+'.mp4'))
    report[name+'_seconds']=time.perf_counter()-t

# The original preview is silent, but its picture stream is copied untouched.
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(ROOT/'input.mp4'),'-map','0:v:0','-c:v','copy','-an','-movflags','+faststart',str(OUT/'input.mp4')],check=True)
media={}
for path in OUT.glob('*.mp4'):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)]))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    media[path.name]={k:stream.get(k) for k in ['codec_name','width','height','avg_frame_rate','nb_read_frames','duration']}
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
report['preview_media']=media
report['rife_model']='v4.26';report['interpolation_factor']=8;report['interpolation_playback_fps']=60
report['runtime_gpu']=torch.cuda.get_device_name(0)
report.pop('output',None)
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print('COMPLETE '+json.dumps(report),flush=True)
(ROOT/'complete.json').write_text(json.dumps({'ok':True,'files':[p.name for p in OUT.iterdir()]}))
