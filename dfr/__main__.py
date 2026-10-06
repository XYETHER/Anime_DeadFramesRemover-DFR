"""Command-line entry point for xyether's DFR pipeline."""
import argparse
import json
import pathlib
import sys

def main():
    parser=argparse.ArgumentParser(description='Anime Dead Frames Remover (DFR), created by xyether')
    parser.add_argument('video',type=pathlib.Path)
    parser.add_argument('--output-dir',type=pathlib.Path,default=pathlib.Path('results'))
    parser.add_argument('--work-dir',type=pathlib.Path,default=pathlib.Path('.dfr-cache'))
    parser.add_argument('--type',choices=['1','2'],default='1',dest='processing_type')
    parser.add_argument('--codec',choices=['h264_nvenc','hevc_nvenc'],default='h264_nvenc')
    args=parser.parse_args()
    if not args.video.is_file():parser.error('Input video does not exist')
    from .pipeline import analyse_video, encode_video
    import cv2, numpy as np, torch
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
    # Reuse the notebook's pinned, checksum-verified setup.
    notebook=pathlib.Path(__file__).resolve().parents[1]/'DFR_Colab.ipynb'
    data=json.loads(notebook.read_text(encoding='utf-8'))
    setup=next(''.join(c['source']) for c in data['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('#@title 1. DFR'))
    setup=setup.replace("pathlib.Path('/content/DFR')",'pathlib.Path('+repr(str(args.work_dir.resolve()))+')')
    namespace={}
    exec(compile(setup,'DFR setup','exec'),namespace)
    from depth_anything_v2.dpt import DepthAnythingV2
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(namespace['DFR_CHECKPOINT'],map_location='cpu',weights_only=True))
    model=model.eval().cuda()
    torch.set_num_threads(2);cv2.setNumThreads(2)
    @torch.inference_mode()
    def depth(frame):
        with torch.autocast('cuda',dtype=torch.float16):
            tensor,_=model.image2tensor(frame,252)
            result=model(tensor)[0]
        result=cv2.resize(result.float().cpu().numpy(),(frame.shape[1],frame.shape[0]))
        if not np.isfinite(result).all():raise RuntimeError('Invalid depth values')
        return result
    import subprocess
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(args.video)]))['streams'][0]
    if stream.get('color_transfer') in ['smpte2084','arib-std-b67']:raise ValueError('SDR input required')
    rows,report=analyse_video(args.video,depth,70,50,args.output_dir,analysis_width=640,frame_step=3 if args.processing_type=='1' else 2)
    report.update(encode_video(args.video,rows,args.output_dir,fps_override='24000/1001',encoder=args.codec))
    report['processing_type']='Type '+args.processing_type
    (args.output_dir/'DFR_report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
