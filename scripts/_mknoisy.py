import os, numpy as np
from scipy.io import wavfile
from scipy import signal
np.random.seed(7)
T=os.environ['TEMP']
sr=16000
sr0,d=wavfile.read("audio_nlp_1_5min.wav")
if d.ndim>1: d=d.mean(axis=1)
clean=(d.astype(np.float64)/32768.0)
fl=int(0.05*sr); n=len(clean)//fl
r=np.sqrt((clean[:n*fl].reshape(n,fl)**2).mean(axis=1))
speech=(r>np.percentile(r,60))

def make(kind, N, rng):
    w=rng.standard_normal(N)
    if kind=="fan":      # AC / fan: low-freq rumble, brown-ish
        b,a=signal.butter(2,300/(sr/2),'low'); return signal.lfilter(b,a,w)*3.0
    if kind=="hiss":     # broadband hiss / mic self-noise
        b,a=signal.butter(1,[400/(sr/2),7000/(sr/2)],'band'); return signal.lfilter(b,a,w)*2.0
    if kind=="hum":      # electrical hum 50Hz + harmonics
        t=np.arange(N)/sr
        return (0.9*np.sin(2*np.pi*50*t)+0.5*np.sin(2*np.pi*100*t)+0.3*np.sin(2*np.pi*150*t)+0.15*np.sin(2*np.pi*200*t))*0.5
    if kind=="babble":   # distant crowd / room
        b,a=signal.butter(2,[200/(sr/2),2500/(sr/2)],'band'); return signal.lfilter(b,a,w)*1.2
    if kind=="mixed":
        return (0.6*make("fan",N,rng)+0.5*make("hiss",N,rng)+0.9*make("hum",N,rng)+0.7*make("babble",N,rng))
N=len(clean)
rng=np.random.default_rng(3)
noise=make("mixed",N,rng)
noise-=noise.mean()
# scale to SNR 8 dB relative to speech-active RMS
sps=np.sqrt((clean[:n*fl].reshape(n,fl)**2).mean(axis=1))[speech].mean()
nps=np.sqrt((noise[:n*fl].reshape(n,fl)**2).mean())
noise*=sps/(10**(8/20))/nps
mix=clean+noise
wavfile.write(T+r"\noisy.wav", sr, np.clip(mix,-1,1).astype(np.float32))
wavfile.write(T+r"\ref_clean.wav", sr, clean.astype(np.float32))
noise_only=(clean*0+noise)
wavfile.write(T+r"\noise_only.wav", sr, noise_only.astype(np.float32))
np.save(T+r"\speechmask.npy", speech)
print("mix peak %.3f  speech_rms %.1f dB  noise_rms %.1f dB  (SNR ~8dB)"%(np.max(np.abs(mix)),20*np.log10(sps),20*np.log10(np.sqrt((noise**2).mean()))))

