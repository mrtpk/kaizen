# https://github.com/s0d3s/PyAudioWPatch
# https://github.com/s0d3s/PyAudioWPatch/blob/master/examples/pawp_record_wasapi_loopback.py

"""A simple example of recording from speakers ('What you hear') using the WASAPI loopback device"""

# from _spinner_helper import Spinner
# Spinner is a helper class that is in the same examples folder.
# It is optional, you can safely delete the code associated with it.

import pyaudiowpatch as pyaudio
import time
import wave

duration = 60.0 * 3

filename = "loopback_record_{}.wav".format(int(duration))
    
    
if __name__ == "__main__":
    # with pyaudio.PyAudio() as p, Spinner() as spinner:
    with pyaudio.PyAudio() as p:
        """
        Create PyAudio instance via context manager.
        Spinner is a helper class, for `pretty` output
        """
        try:
            # Get default WASAPI info
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("Looks like WASAPI is not available on the system. Exiting...")
            # spinner.print("Looks like WASAPI is not available on the system. Exiting...")
            # spinner.stop()
            exit()
        
        # Get default WASAPI speakers
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
                # spinner.print("Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
                # spinner.stop()
                exit()
                
        print(f"Recording from: ({default_speakers['index']}){default_speakers['name']}")
        # spinner.print(f"Recording from: ({default_speakers['index']}){default_speakers['name']}")
        
        wave_file = wave.open(filename, 'wb')
        wave_file.setnchannels(default_speakers["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(default_speakers["defaultSampleRate"]))

        
        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            wave_file.writeframes(in_data)
            print(type(in_data), frame_count, time_info, status)
            return (in_data, pyaudio.paContinue)
        
        with p.open(format=pyaudio.paInt16,
                channels=default_speakers["maxInputChannels"],
                rate=int(default_speakers["defaultSampleRate"]),
                # frames_per_buffer=pyaudio.get_sample_size(pyaudio.paInt8),
                frames_per_buffer=pyaudio.get_sample_size(pyaudio.paInt16),
                # frames_per_buffer=pyaudio.get_sample_size(pyaudio.paInt16),
                # frames_per_buffer=int(default_speakers["defaultSampleRate"]),
                input=True,
                input_device_index=default_speakers["index"],
                stream_callback=callback
        ) as stream:
            """
            Opena PA stream via context manager.
            After leaving the context, everything will
            be correctly closed(Stream, PyAudio manager)            
            """
            # spinner.print(f"The next {duration} seconds will be written to {filename}")
            print(f"The next {duration} seconds will be written to {filename}")
            time.sleep(duration) # Blocking execution while playing
        
        wave_file.close()
