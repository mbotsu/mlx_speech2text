import os
import sys
import argparse
import mlx_whisper
from mlx_whisper.transcribe import _format_timestamp
from writers import get_writer
from silero_vad import (
    load_silero_vad,
    get_speech_timestamps,
    read_audio,
    collect_chunks
)

SAMPLING_RATE = 16000
SPEECH_PAD_MS = 400
WHISPER_MODEL='mlx-community/whisper-turbo'

def main(speech_file, output_dir, language, verbose, threshold, write_file="out"):
    wav = read_audio(speech_file, sampling_rate=SAMPLING_RATE)
    model = load_silero_vad()
    speech_timestamps = get_speech_timestamps(wav, model,
                                            sampling_rate=SAMPLING_RATE,
                                            return_seconds=False,
                                            min_silence_duration_ms=2000)

    result = {
        'segments': []
    }

    system_encoding = sys.getdefaultencoding()
    if system_encoding != "utf-8":
        make_safe = lambda x: x.encode(system_encoding, errors="replace").decode(
            system_encoding
        )
    else:
        make_safe = lambda x: x

    wav_length = len(wav)
    for seek in speech_timestamps:
        seek_start = seek['start']
        if seek_start <= 0:
            seek_start = 0
        
        seek_end = seek['end']
        if seek_end >= wav_length:
            seek_end = wav_length
        
        wav_slice = wav[seek_start:seek_end]
        start = round(seek['start'] / SAMPLING_RATE, 1)
        end = round(seek['end'] / SAMPLING_RATE, 1)
        speech_timestamps2 = get_speech_timestamps(wav_slice, model,
                                                    sampling_rate=SAMPLING_RATE,
                                                    speech_pad_ms=SPEECH_PAD_MS,
                                                    threshold=threshold,
                                                    return_seconds=False)
        if len(speech_timestamps2) == 0:
            continue

        audio = collect_chunks(speech_timestamps2, wav_slice)
        audio = audio.to('cpu').detach().numpy().copy()

        trans = mlx_whisper.transcribe(audio, verbose=False, language=language,
                                    path_or_hf_repo=WHISPER_MODEL, fp16=True,
                                    no_speech_threshold=0.1,
                                    compression_ratio_threshold=1.4,
                                    logprob_threshold=-1)

        for r in trans['segments']:
            if r['text'] == "":
                continue
            res = {
                'start': start + r['start'],
                'end': start + r['end'],
                'text': r['text'],
            }
            if verbose:
                result['segments'].append(res)
                line = f"[{_format_timestamp(res['start'])} --> {_format_timestamp(res['end'])}] {res['text']}"
                print(make_safe(line))        
        
    writer_args = {'highlight_words': False, 'max_line_count': None, 'max_line_width': None, 'pretty_json': False}

    os.makedirs(output_dir, exist_ok=True)
    writer = get_writer("json", output_dir)
    writer(result, write_file, writer_args)

    writer = get_writer("vtt", output_dir)
    writer(result, write_file, writer_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcription for Apple Silicon.')
    parser.add_argument('-i', '--input', help='input audio file', required=True)
    parser.add_argument('-o', '--output', default="track", help='output text folder')
    parser.add_argument('-l', '--language', default="ja", help='language')
    parser.add_argument('-v', '--varbose', action='store_true', help='varbose')
    parser.add_argument('-t', '--threshold', default=0.5, help='silero-vad threshold', type=float)    
    args = parser.parse_args()

    main(args.input, args.output, args.language, args.varbose, args.threshold, write_file="out")
