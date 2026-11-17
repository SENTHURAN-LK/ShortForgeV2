from __future__ import annotations
import argparse, json, math, re, shutil, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
OUT, WORK = ROOT / 'output', ROOT / 'work'
@dataclass
class Word:
    text: str
    start: float
    end: float
@dataclass
class Candidate:
    start: float
    end: float
    text: str
    score: float
    reasons: list[str]

HOOKS = {'why','how','secret','truth','mistake','never','actually','imagine','problem','discovered','instead','because','impossible','first','worst','best','surprising','strange','hidden','forgotten','until','revealed','realized','turns','however','but'}
FILLERS = {'uh','um','erm','you','know','like','okay','so'}
WEAK_START = {'and','but','so','because','uh','um','then','which','that'}


def run(cmd: list[str], cwd: Optional[Path] = None):
    print('+', subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

def require(exe: str):
    if not shutil.which(exe): raise RuntimeError(f'{exe} not found on PATH. Install it, then reopen PowerShell.')

def download_video(url: str) -> Path:
    from yt_dlp import YoutubeDL
    WORK.mkdir(parents=True, exist_ok=True)
    opts={'format':'bv*[height<=1080]+ba/b[height<=1080]/b','merge_output_format':'mp4','outtmpl':str(WORK/'source.%(ext)s'),'noplaylist':True}
    with YoutubeDL(opts) as ydl:
        info=ydl.extract_info(url, download=True); path=Path(ydl.prepare_filename(info))
        if path.suffix.lower() != '.mp4' and path.with_suffix('.mp4').exists(): path=path.with_suffix('.mp4')
        if not path.exists():
            found=list(WORK.glob('source.*'))
            if not found: raise FileNotFoundError('Downloaded video was not found.')
            path=found[0]
    return path

def transcribe(video: Path, model_name='small', device='auto') -> list[Word]:
    from faster_whisper import WhisperModel
    chosen_device='cuda' if device in ('auto','cuda') else 'cpu'
    compute='float16' if chosen_device=='cuda' else 'int8'
    try:
        model=WhisperModel(model_name, device=chosen_device, compute_type=compute)
    except Exception as e:
        if device=='cuda': raise RuntimeError(f'CUDA transcription failed: {e}. Check NVIDIA driver, CUDA 12 cuBLAS and cuDNN 9.') from e
        print(f'GPU unavailable ({e}); falling back to CPU int8.')
        model=WhisperModel(model_name, device='cpu', compute_type='int8')
    segments,_=model.transcribe(str(video), word_timestamps=True, vad_filter=True, condition_on_previous_text=False, beam_size=5)
    words=[]
    for seg in segments:
        for w in (seg.words or []):
            if w.start is not None and w.end is not None and w.word.strip(): words.append(Word(w.word.strip(),float(w.start),float(w.end)))
    if not words: raise RuntimeError('No speech detected in this video.')
    WORK.mkdir(parents=True,exist_ok=True)
    (WORK/'transcript.json').write_text(json.dumps([w.__dict__ for w in words],indent=2),encoding='utf-8')
    return words

def tokenise(s: str) -> list[str]: return re.findall(r"[a-z0-9']+",s.lower())
def make_candidates(words: list[Word], min_len=15.0, max_len=30.0, max_clips=8) -> list[Candidate]:
    """Rank coherent, self-contained speech windows using multi-signal editorial scoring.
    This is an explainable local ranker, not a claim of human-level semantic understanding.
    """
    if not words: return []
    # Build candidate starts at sentence boundaries plus a rolling fallback every ~3 sec.
    starts=set()
    for i,w in enumerate(words):
        if i==0 or words[i-1].text.endswith(('.', '?','!')): starts.add(i)
    last_roll = -1e9
    for i,w in enumerate(words):
        if w.start - last_roll >= 3.0:
            starts.add(i)
            last_roll = w.start
    scored=[]
    for i in sorted(starts):
        start=words[i].start
        for j in range(i+1,len(words)):
            end=words[j].end; dur=end-start
            if dur>max_len: break
            if dur<min_len: continue
            chunk=words[i:j+1]; text=' '.join(x.text for x in chunk); toks=tokenise(text)
            if len(toks)<18: continue
            hook=sum(t in HOOKS for t in toks)
            density=len(toks)/max(dur,1)
            density_score=max(0, 1.8-abs(density-2.35)*0.75)
            start_sentence=(i==0 or words[i-1].text.endswith(('.', '?','!')))
            end_sentence=bool(re.search(r'[.!?][\"\']?$',text))
            question= '?' in text
            story_markers=sum(t in {'when','then','after','before','until','finally','suddenly','realized','decided'} for t in toks)
            contrast=sum(t in {'but','however','although','instead','yet','except'} for t in toks)
            specifics=sum(bool(re.fullmatch(r'\d+(?:\.\d+)?',t)) for t in toks)
            first_person=sum(t in {'i','we','my','our','me'} for t in toks)
            filler=sum(t in FILLERS for t in toks)
            weak=toks[0] in WEAK_START
            # Weighted editorial signals. Sentence-complete boundaries and specificity help clips stand alone.
            score=(min(hook,5)*0.75 + density_score + (1.3 if start_sentence else 0) + (1.8 if end_sentence else 0)
                   + min(story_markers,4)*0.35 + min(contrast,3)*0.45 + min(specifics,3)*0.3
                   + (0.45 if question else 0) + (0.2 if first_person else 0)
                   - min(filler,8)*0.12 - (1.2 if weak else 0))
            reasons=[]
            if hook: reasons.append('hook language')
            if end_sentence: reasons.append('complete ending')
            if story_markers: reasons.append('story progression')
            if contrast: reasons.append('contrast or reveal')
            if specifics: reasons.append('specific detail')
            scored.append(Candidate(start,end,text,round(score,3),reasons))
    scored.sort(key=lambda c:c.score,reverse=True)
    chosen=[]
    for c in scored:
        # Prevent near-duplicate clips; 6-second breathing room.
        if all(c.end <= x.start-6 or c.start >= x.end+6 for x in chosen): chosen.append(c)
        if len(chosen)>=max_clips: break
    return sorted(chosen,key=lambda c:c.start)

def ass_time(seconds: float) -> str:
    cs=max(0,round(seconds*100)); h,rem=divmod(cs,360000); m,rem=divmod(rem,6000); s,cs=divmod(rem,100)
    return f'{h}:{m:02d}:{s:02d}.{cs:02d}'
def ass_escape(s: str) -> str: return s.replace('\\',r'\\').replace('{',r'\\{').replace('}',r'\\}')
def write_ass(words: list[Word], start: float, end: float, path: Path):
    selected=[w for w in words if w.end>start and w.start<end]
    lines=['[Script Info]','ScriptType: v4.00+','PlayResX: 1080','PlayResY: 1920','WrapStyle: 2','','[V4+ Styles]',
      'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
      'Style: Default,Arial,72,&H00FFFFFF,&H0000D7FF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,75,75,370,1','','[Events]','Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
    # 4-word groups make captions easier to read on mobile; karaoke tags animate each word.
    for k in range(0,len(selected),4):
        group=selected[k:k+4]; a=max(start,group[0].start); b=min(end,group[-1].end+0.18); parts=[]
        for w in group:
            dur=max(1,round((w.end-w.start)*100)); parts.append(r'{\k%d}%s'%(dur,ass_escape(w.text)))
        lines.append(f'Dialogue: 0,{ass_time(a-start)},{ass_time(b-start)},Default,,0,0,0,,{" ".join(parts)}')
    path.write_text('\n'.join(lines),encoding='utf-8-sig')

def detect_subject_centres(video: Path, start: float, end: float, sample_fps=1.5) -> list[tuple[float,float]]:
    """Detect people at sparse intervals and return (time, normalized x-centre).
    Requires ultralytics; falls back cleanly to centre crop if unavailable/no person.
    """
    try:
        import cv2
        from ultralytics import YOLO
    except Exception as e:
        print(f'Smart crop unavailable ({e}); using centre crop. Install ultralytics + CUDA PyTorch to enable it.')
        return []
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): return []
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0; width=cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
    model=YOLO('yolo26n.pt'); samples=[]; t=start; previous=None
    while t<end:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read()
        if not ok: break
        result=model.predict(frame,classes=[0],conf=0.25,verbose=False,device=0 if __import__('torch').cuda.is_available() else 'cpu')[0]
        boxes=result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        candidates=[]
        for x1,y1,x2,y2 in boxes:
            cx=((x1+x2)/2)/width; area=((x2-x1)*(y2-y1))/(width*height)
            # Prefer a large, central subject; continuity discourages switching people.
            continuity=abs(cx-previous) if previous is not None else abs(cx-0.5)
            candidates.append((area*2.0-continuity*0.9,cx))
        if candidates:
            cx=max(candidates)[1]; previous=cx; samples.append((t,cx))
        t+=1.0/sample_fps
    cap.release()
    return samples

def crop_expression(samples: list[tuple[float,float]], clip_start: float) -> str:
    # FFmpeg crop x expression interpolates between detected subject centres; fallback centre.
    if not samples: return 'iw/2-ih*9/16/2'
    # Expression kept compact; sample x positions are normalized to input width.
    terms=[]
    for t,cx in samples:
        rel=max(0,t-clip_start); terms.append((rel,cx))
    # Use piecewise linear interpolation through nested if/lerp-like expressions.
    expr=f'(iw/2-ih*9/16/2)'
    for idx,(t,cx) in enumerate(reversed(terms)):
        target=f'(iw*{cx:.5f}-ih*9/16/2)'
        if idx==0: expr=target
        else:
            pt,pc= list(reversed(terms))[len(terms)-idx]
            prev=f'(iw*{pc:.5f}-ih*9/16/2)'
            expr=f'if(lt(t,{t:.3f}),{prev}+({target}-{prev})*(t-{pt:.3f})/{max(t-pt,0.001):.3f},{expr})'
    return f'min(max({expr},0),iw-ih*9/16)'

def render_crop(video: Path, candidate: Candidate, idx: int, smart_crop=True) -> Path:
    # Use subject-aware detection if available; render an intermediate crop then burn captions.
    samples=detect_subject_centres(video,candidate.start,candidate.end) if smart_crop else []
    meta=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',str(video)],text=True)
    dims=json.loads(meta)['streams'][0]; w,h=int(dims['width']),int(dims['height'])
    crop_w=int(h*9/16); crop_w=min(crop_w,w); crop_w-=crop_w%2; h2=h-h%2
    # For each frame, Python OpenCV applies interpolated crop coordinates. Robust and smooth, albeit slower than GPU encoding.
    try: import cv2
    except Exception: samples=[]
    if not samples:
        # FFmpeg centre crop path (fast, no OpenCV dependency).
        raw=WORK/f'crop_{idx:02d}.mp4'
        vf=f'scale=-2:{h2},crop={int(h2*9/16)//2*2}:{h2},scale=1080:1920'
        run(['ffmpeg','-y','-ss',f'{candidate.start:.3f}','-i',str(video),'-t',f'{candidate.end-candidate.start:.3f}','-vf',vf,'-an','-c:v','libx264','-preset','fast','-crf','19',str(raw)])
        return raw
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC,candidate.start*1000)
    raw=WORK/f'crop_{idx:02d}.mp4'; writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1080,1920))
    frame_count=int((candidate.end-candidate.start)*fps); times=[x[0] for x in samples]; centres=[x[1] for x in samples]
    import numpy as np
    for n in range(frame_count):
        ok,frame=cap.read()
        if not ok: break
        now=candidate.start+n/fps; cx=float(np.interp(now,times,centres)); x=int(cx*frame.shape[1]-crop_w/2); x=max(0,min(frame.shape[1]-crop_w,x))
        y=0; crop=frame[y:y+h2,x:x+crop_w]
        if crop.size: writer.write(cv2.resize(crop,(1080,1920),interpolation=cv2.INTER_AREA))
    cap.release(); writer.release()
    return raw

def render_clips(video: Path, words: list[Word], candidates: list[Candidate], smart_crop=True) -> list[Path]:
    OUT.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True); rendered=[]
    for idx,c in enumerate(candidates,1):
        ass=WORK/f'clip_{idx:02d}.ass'; write_ass(words,c.start,c.end,ass)
        raw=render_crop(video,c,idx,smart_crop=smart_crop); target=OUT/f'short_{idx:02d}.mp4'
        # Captions are timed relative to clip start. Source audio is trimmed separately and muxed back.
        escaped=str(ass.resolve()).replace('\\','/').replace(':','\\:').replace("'",r"\'")
        run(['ffmpeg','-y','-i',str(raw),'-ss',f'{c.start:.3f}','-i',str(video),'-t',f'{c.end-c.start:.3f}',
             '-vf',f"subtitles='{escaped}'",'-map','0:v:0','-map','1:a?','-c:v','libx264','-preset','medium','-crf','20','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',str(target)])
        rendered.append(target)
        print(f'[{idx}/{len(candidates)}] {target.name} {c.start:.1f}-{c.end:.1f}s | score={c.score} | {", ".join(c.reasons)}')
    return rendered

def import_into_resolve(files: list[Path]):
    try: import DaVinciResolveScript as dvr
    except Exception as e: raise RuntimeError('Resolve scripting module not found. Configure PYTHONPATH from Resolve Developer documentation.') from e
    resolve=dvr.scriptapp('Resolve')
    if not resolve: raise RuntimeError('Could not connect to Resolve. Open Resolve and enable external scripting access.')
    project=resolve.GetProjectManager().GetCurrentProject()
    if not project: raise RuntimeError('Open a Resolve project first.')
    pool=project.GetMediaPool(); clips=pool.ImportMedia([str(p.resolve()) for p in files])
    if not clips: raise RuntimeError('Resolve did not import clips.')
    timeline=pool.CreateTimelineFromClips('ShortForge V2 - Auto Shorts',clips)
    if not timeline: raise RuntimeError('Imported media, but timeline creation failed.')
    print('Resolve timeline created:',timeline.GetName())

def main():
    p=argparse.ArgumentParser(description='ShortForge V2: automatic YouTube-to-Shorts pipeline')
    p.add_argument('url'); p.add_argument('--resolve',action='store_true'); p.add_argument('--max-clips',type=int,default=8)
    p.add_argument('--whisper-model',default='small',choices=['tiny','base','small','medium','large-v3'])
    p.add_argument('--device',default='auto',choices=['auto','cuda','cpu']); p.add_argument('--no-smart-crop',action='store_true')
    a=p.parse_args(); require('ffmpeg'); require('ffprobe'); WORK.mkdir(parents=True,exist_ok=True)
    video=download_video(a.url); words=transcribe(video,a.whisper_model,a.device)
    candidates=make_candidates(words,max_clips=max(1,min(a.max_clips,12)))
    if not candidates: raise RuntimeError('No suitable 15–30 second speech segments found.')
    (OUT/'clips.json').write_text(json.dumps([c.__dict__ for c in candidates],indent=2),encoding='utf-8')
    files=render_clips(video,words,candidates,smart_crop=not a.no_smart_crop)
    if a.resolve: import_into_resolve(files)
    print(f'Finished: {len(files)} shorts saved to {OUT.resolve()}')
if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print('Cancelled.'); sys.exit(130)
    except Exception as e: print(f'ERROR: {e}',file=sys.stderr); sys.exit(1)
