# https://github.com/s0d3s/PyAudioWPatch
# https://github.com/s0d3s/PyAudioWPatch/blob/master/examples/pawp_record_wasapi_loopback.py


# https://stackoverflow.com/a/36116237/6561141

"""A simple example of recording from speakers ('What you hear') using the WASAPI loopback device"""

# from _spinner_helper import Spinner
# Spinner is a helper class that is in the same examples folder.
# It is optional, you can safely delete the code associated with it.

import pyaudiowpatch as pyaudio
import time
import wave
import numpy as np

duration = 60.0 * 3

filename = "loopback_record_{}.wav".format(int(duration))

FRAMES_PER_BUFFER = pyaudio.get_sample_size(pyaudio.paInt16)
FRAMES_PER_BUFFER = 48000

COUNTER = 0
AUDIO_DATA = None
IS_LOCKED = False

def encode(audio_data, num_levels):
    global IS_LOCKED
    result = 0
    while(IS_LOCKED):
        pass

    IS_LOCKED = True
    channel_one_data = np.copy(audio_data[0, ...])
    IS_LOCKED = False

    # normed = channel_one_data
    # min_value = np.iinfo(channel_one_data.dtype).min # https://stackoverflow.com/a/23189557/6561141 - iinfo works only for int
    # max_value = np.iinfo(channel_one_data.dtype).max

    normed = np.abs(channel_one_data)
    min_value = normed.min()
    max_value = normed.max()

    normed = (normed - min_value) / (max_value - min_value)
    # normed = np.abs(normed)

    projected = np.ceil(normed * num_levels).astype(np.int)
    counter = np.unique(projected, return_counts=True)
    # print(counter)
    values, counts = counter
    max_idx = np.argmax(counts)
    result = values[max_idx]
    return result
    
if __name__ == "__main__":
    with pyaudio.PyAudio() as p:
        """
        Create PyAudio instance via context manager.
        """
        # Get default WASAPI info
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                """
                Try to find loopback device with same name(and [Loopback suffix]).
                Unfortunately, this is the most adequate way at the moment.
                """
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print("Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
                exit()

        # import pdb; pdb.set_trace()

        print(f"Recording from: ({default_speakers['index']}){default_speakers['name']}")
        
        wave_file = wave.open(filename, 'wb')
        wave_file.setnchannels(default_speakers["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(default_speakers["defaultSampleRate"]))

        
        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            global COUNTER, AUDIO_DATA, IS_LOCKED
            COUNTER = COUNTER + 1

            wave_file.writeframes(in_data) # save to the wave file for debug
            

            if not IS_LOCKED:
                IS_LOCKED = True
                audio_chunk = np.frombuffer(in_data, dtype=np.int16) # audio in interleaved channels
                audio_chunk = np.stack((audio_chunk[::2], audio_chunk[1::2]), axis=0)
                AUDIO_DATA = audio_chunk
                IS_LOCKED = False
            # print(audio_chunk.shape)
            # print(COUNTER, "-->", len(in_data), frame_count, time_info)
            return (in_data, pyaudio.paContinue)
        
        with p.open(format=pyaudio.paInt16,
                channels=default_speakers["maxInputChannels"],
                rate=int(default_speakers["defaultSampleRate"]),
                frames_per_buffer=FRAMES_PER_BUFFER,
                input=True,
                input_device_index=default_speakers["index"],
                stream_callback=callback
        ) as stream:

            # PREV_AUDIO_DATA = AUDIO_DATA
            # while(True):
            #     if AUDIO_DATA is not None:
            #         output_signal = encode(AUDIO_DATA, num_levels=32)
            #         print(output_signal)

            #         # if PREV_AUDIO_DATA is not None and np.all(PREV_AUDIO_DATA == AUDIO_DATA):
            #         #     break
            #         PREV_AUDIO_DATA = AUDIO_DATA
            time.sleep(duration) # Blocking execution while playing
            

        wave_file.close()
