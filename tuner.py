import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SAMPLE_RATE = 44100
CHUNK_SIZE = 4096

latest_audio = np.zeros(CHUNK_SIZE)

# note names and their frequencies (A4 = 440 Hz standard tuning)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note(freq):
    if freq <= 0:
        return "---"
    # convert frequency to MIDI note number, then to note name
    midi = 12 * np.log2(freq / 440.0) + 69
    midi_rounded = int(round(midi))
    octave = (midi_rounded // 12) - 1
    note = NOTE_NAMES[midi_rounded % 12]
    return f"{note}{octave}"

def callback(indata, frames, time, status):
    global latest_audio
    latest_audio = indata[:, 0]

fig, ax = plt.subplots()
x = np.fft.rfftfreq(CHUNK_SIZE, d=1/SAMPLE_RATE)
line, = ax.plot(x, np.zeros(len(x)))
ax.set_xlim(0, 5000)
ax.set_ylim(0, 0.01)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")

# text display for the detected note
note_text = ax.text(0.05, 0.75, "Note: ---", transform=ax.transAxes, fontsize=20, fontweight='bold', color='red')

def update(frame):
    fft = np.abs(np.fft.rfft(latest_audio)) / CHUNK_SIZE

    # only detect notes in guitar frequency range (80Hz - 1200Hz)
    low = int(80 * CHUNK_SIZE / SAMPLE_RATE)
    high = int(1200 * CHUNK_SIZE / SAMPLE_RATE)
    fft_guitar = fft[low:high]

    # find the peak frequency
    peak_index = np.argmax(fft_guitar) + low
    peak_freq = peak_index * SAMPLE_RATE / CHUNK_SIZE

    # only show a note if signal is strong enough (filters out silence)
    if np.max(fft_guitar) > 0.001:
        note = freq_to_note(peak_freq)
        
        # calculate cents offset from perfect pitch
        midi_exact = 12 * np.log2(peak_freq / 440.0) + 69
        midi_rounded = round(midi_exact)
        cents = (midi_exact - midi_rounded) * 100

        # color code: green = in tune, red = out of tune
        if abs(cents) < 5:
            note_text.set_color('green')
            tuning = "IN TUNE"
        elif cents < 0:
            note_text.set_color('red')
            tuning = f"FLAT {abs(cents):.0f}¢"
        else:
            note_text.set_color('red')
            tuning = f"SHARP {cents:.0f}¢"

        note_text.set_text(f"Note: {note}  ({peak_freq:.1f} Hz)\n{tuning}")
    else:
        note_text.set_color('white')
        note_text.set_text("Note: ---")
        

    line.set_ydata(fft)
    return line, note_text

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, callback=callback)

with stream:
    ani = animation.FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
    plt.show()