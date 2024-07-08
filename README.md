## Abstract
Transcription for Apple Silicon

## Install
```
$ git clone https://github.com/mbotsu/mlx_speech2text.git
$ pip install -r requirements.txt
$ mkdir models
$ curl -L https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.jit -o models/silero_vad.jit 
```

## RUN
```
// convert to wav 16K
$ ffmpeg -i input.mp4 -ar 16000 out.wav

// run
$ python speech2text.py -i out.wav -o track
```

## References
- [ml-explore/mlx-examples/whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- [Softcatala/whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2)
- [Segmenting a long audio file #295](https://github.com/snakers4/silero-vad/discussions/295)
