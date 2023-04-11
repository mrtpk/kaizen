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
import math
import cv2

from color import get_color_samples, get_static_color_samples

SR = 48000
NUM_BULBS = 4

MIN_DB = -50
MAX_DB = -5

# duration = 60.0 * 3
# LIMIT = int((duration * SR) / FRAMES_PER_BUFFER)

LIMIT = 2**1024

# filename = "loopback_record_{}.wav".format(int(duration))


NUM_LEVELS = NUM_BULBS + 1
DB_LENGTH = (MAX_DB - MIN_DB)

RATIO = (MIN_DB/MAX_DB) ** (1 / (NUM_LEVELS - 1)) # https://www.geeksforgeeks.org/how-to-find-common-ratio-with-first-and-last-terms/
INTERVAL_LIMITS = [np.floor(MAX_DB * (RATIO ** x)) for x in range(0, NUM_LEVELS)] # https://math.stackexchange.com/a/3761117/516715
# print(INTERVAL_LIMITS)

# FRAMES_PER_BUFFER = SR
FRAMES_PER_BUFFER = int(SR * (1 / 300)) # number of samples in 300 ms # why 300 ms? https://emastered.com/blog/rms-level-for-mastering



COUNTER = 0
AUDIO_DATA = None
IS_LOCKED = False

# init colors
COLORS = {}
try:
    color_samples = get_static_color_samples(num_color_samples=NUM_LEVELS)
    print("took colors from static")
except Exception as err:
    color_samples = get_color_samples(num_color_samples=NUM_LEVELS)
for idx, color_sample in zip(range(0, NUM_LEVELS), color_samples):
    r, g, b = color_sample
    COLORS[idx+1] = (b, g, r)

def encode(audio_data, num_levels):
    global IS_LOCKED
    result = 0
    while(IS_LOCKED):
        pass

    IS_LOCKED = True
    channel_one_data = np.copy(audio_data[0, ...])
    IS_LOCKED = False
    
    channel_one_data = channel_one_data.astype(np.float32) * (1.0 / 32768.0) # 16 bit PCM has a range - 32768 to 32767. So, multiply each of your PCM samples by (1.0f/32768.0f) # https://stackoverflow.com/a/15091042/6561141

    block_linear_rms = np.sqrt(np.mean(channel_one_data**2)) # Linear value between 0 -> 1
    block_log_rms = np.floor(20 * math.log10(block_linear_rms)) # Decibel (dB value) between 0 dB -> -inf dB

    # block_log_rms = np.clip(block_log_rms, a_min=MAX_DB, a_max=MIN_DB) # clip the db

    result = np.digitize(x=block_log_rms, bins=INTERVAL_LIMITS, right=False) # https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
    # print(INTERVAL_LIMITS, block_log_rms, result) 
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

        print(f"Showing from: ({default_speakers['index']}){default_speakers['name']}")
        
        # wave_file = wave.open(filename, 'wb')
        # wave_file.setnchannels(default_speakers["maxInputChannels"])
        # wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        # wave_file.setframerate(int(default_speakers["defaultSampleRate"]))

        
        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            global COUNTER, AUDIO_DATA, IS_LOCKED
            COUNTER = COUNTER + 1

            # wave_file.writeframes(in_data) # save to the wave file for debug
            

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

            PREV_AUDIO_DATA = AUDIO_DATA
            counter = 0

            while(counter < LIMIT):
                if AUDIO_DATA is not None:
                    output_signal = encode(AUDIO_DATA, num_levels=NUM_LEVELS)
                    # print(output_signal, end=" ")
                    img = np.ones([720, 1280, 3], dtype=np.uint8) * COLORS[output_signal]
                    img = img.astype(np.uint8)  
                    cv2.imshow('music in color', img)
                    cv2.waitKey(1)

                    # if PREV_AUDIO_DATA is not None and np.all(PREV_AUDIO_DATA == AUDIO_DATA):
                    #     break
                    PREV_AUDIO_DATA = AUDIO_DATA
                    counter = counter + 1
            # time.sleep(duration) # Blocking execution while playing
            

        # wave_file.close()
        # import pdb; pdb.set_trace()