from tkinter import *
from tkinter import ttk
import pyaudio
import wave
from sympy import symbols
import numpy as np
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from scipy import signal  # this is imported just to take fast fourier convolution.

app = Tk()
app.title('Main App')
app.geometry('700x500')

z = symbols('z')
Label(app, text='Recording and Playing-').grid(row=0, column=0, pady=15, padx=10)
Label(app, text='Choose any option :').grid(row=1, column=0, padx=15)

recPlay = IntVar()
recPlay.set(1)

ttk.Radiobutton(app, text="Record and Play", variable=recPlay, value=1,
                command=lambda: rec_option()).grid(row=1, column=1)
ttk.Radiobutton(app, text="Record, Save and Play", variable=recPlay, value=2,
                command=lambda: rec_option()).grid(row=1, column=2)


def rec_option():
    if recPlay.get() == 2:
        bSave.config(state='normal')
        bPlay.config(state='normal')
        bRec.config(text='Record', command=clickRec)
    else:
        bSave.config(state='disabled')
        bPlay.config(state='disabled')
        bRec.config(text='Record and Play', command=clickrecANDplay)


Label(app, text='Form of H(z) :').grid(row=3, column=0, pady=25)


def hzform():
    if formHz.get() == "pz":
        lblb.configure(text='Enter zeros of H(z) :')
        lbla.configure(text='Enter poles of H(z) :')
    elif formHz.get() == "lccde":
        lblb.configure(text='Enter coefficients of b :')
        lbla.configure(text='Enter coefficients of a :')
    else:
        lblb.configure(text='Enter coefficients of h[n] :')


formHz = StringVar()
ttk.Radiobutton(app, text='Pole/Zero form', variable=formHz,
                value="pz", command=lambda: hzform()).grid(row=3, column=1)
ttk.Radiobutton(app, text='LCCDE coefficients', variable=formHz,
                value="lccde", command=lambda: hzform()).grid(row=3, column=2)
ttk.Radiobutton(app, text='h[n] form', variable=formHz,
                value="hnform", command=lambda: hzform()).grid(row=3, column=3)

lblb = Label(app, text='Enter coefficients of b :')
lblb.grid(row=4, column=0)

lbla = Label(app, text='Enter coefficients of a :')
lbla.grid(row=5, column=0)
ap = StringVar()
bz = StringVar()

Entry(app, textvariable=bz, width=50).grid(row=4, column=1, columnspan=5, padx=15)
Entry(app, textvariable=ap, width=50).grid(row=5, column=1, columnspan=5, padx=15)

Label(app, text="Enter the type of filter :").grid(row=6, column=0)

fill = StringVar()
filterType = ttk.Combobox(app, values=['LP', 'HP', 'BP', 'BS'],
                          width=5, textvariable=fill)
filterType.grid(row=6, column=1, padx=38, pady=20, sticky='w')
filterType.current(0)

Label(app, text='Enter cutoff frequency range :').grid(row=7, column=0)
freqLow = StringVar()
freqHigh = StringVar()
Entry(app, textvariable=freqLow, width=8).grid(row=7, column=1)
Entry(app, textvariable=freqHigh, width=8).grid(row=7, column=2)

# ****************  the main code for filtering starts here  *******************

WAVE_OUTPUT_FILENAME = "C:\\Users\\Comtech\\Desktop\\output.wav"
CHUNK = 1024  # number of frames the signal is split into
FORMAT = pyaudio.paFloat32  # 2 byte for each sample
CHANNELS = 1  # number of samples contained in
# each frame as the recording is in mono
SAMPLING_RATE = 48000
RECORD_SECONDS = 5  # duration of recording

frames = []  # the array in which audio signal will be stored

global hn
hn = np.array([])


# **************  code for extracting Hz  **************
def process():
    lb = np.array([])
    la = np.array([])
    Hz = 1
    tempStr1 = list(bz.get().split(' '))
    # df = pd.DataFrame(tempStr1)
    # df["id"] = df['id'].tempstr1.replace(' ', '').astype(float)
    i = 0
    while i < len(tempStr1):
        # print(len(tempStr1[i]), tempStr1[i])
        t = float(tempStr1[i])
        np.append(lb, t)
        i += 1

    tempStr2 = list(ap.get().split(' '))
    i = 0
    while i < len(tempStr2):
        # print(len(tempStr2[i]), tempStr2[i])
        t = float(tempStr2[i])
        np.append(la, t)
        i += 1

    if formHz.get() == "pz":
        for rb in lb:
            Hz = Hz * (z - rb)
        for ra in la:
            Hz = Hz / (z - ra)
    elif formHz.get() == "lccde":
        num = np.poly1d(lb)
        den = np.poly1d(la)
        rarrb = num.roots
        rarra = den.roots
        for rb in rarrb:
            Hz = Hz * (z - rb)
        for ra in rarra:
            Hz = Hz / (z - ra)

    # ************* according to type of filter ****************

    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    n = np.arange(N)
    Fs = 1000
    if not N % 2:
        N += 1  # Make sure that N is odd.

    def evalHz(freq):
        # print(freq)
        ff = int(freq)
        func = lambdify(z, Hz, 'numpy')
        print(Hz)
        k = np.linspace(1, ff, ff)
        H = func(np.exp(1j * (2*np.pi*k)/ff))
        H = H / np.amax(H)
        return H

    def applyWindow(h_n):
        # Compute Blackman window.
        w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
            0.08 * np.cos(4 * np.pi * n / (N - 1))

        # Multiply filter by window.
        h_n = h_n * w

        # Normalize to get unity gain.
        h_n = h_n / np.sum(h_n)
        return h_n

    global hn
    hn = np.array([])
    if fill.get() == 'LP':
        fc = float(freqLow.get()) # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
        if formHz.get() != 'hnform':
            H = evalHz(fc)
            # H = np.fft.fftshift(H)
            h_n = np.real(np.fft.ifft(H))
            hn = np.append(hn, h_n)
        else:
            hn = np.append(hn, lb)
            # print(hn)
        hn = applyWindow(hn)

    elif fill.get() == 'HP':
        fc = float(freqLow.get()) / Fs  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
        if formHz.get() != 'hnform':
            H = evalHz(fc)
            h_n = np.real(np.fft.ifft(H))
            hn = np.append(hn, h_n)
        else:
            hn = np.append(hn, lb)
            # print(hn)
        hn = applyWindow(hn)
        hn = -hn
        hn[(N - 1) // 2] += 1

    elif fill.get() == 'BP':
        fL = float(freqLow.get()) / Fs
        fH = float(freqHigh.get()) / Fs

        hL = np.array([])
        hH = np.array([])

        if formHz.get() != 'hnform':
            H = evalHz(fL)
            h_n = np.real(np.fft.ifft(H))
            hL = np.append(hL, h_n)
        else:
            hL = np.append(hL, lb)
            # print(hn)

        if formHz.get() != 'hnform':
            H = evalHz(fH)
            h_n = np.real(np.fft.ifft(H))
            hH = np.append(hH, h_n)
        else:
            hH = np.append(hH, lb)
            # print(hn)

        hL = applyWindow(hL)
        hH = applyWindow(hH)

        hH = -hH
        hH[N - 1 // 2] += 1
        h_n = signal.fftconvolve(hL, hH, 'full')
        hn = np.append(hn, h_n)

    else:
        fL = float(freqLow.get()) / Fs
        fH = float(freqHigh.get()) / Fs
        hL = np.array([])
        hH = np.array([])

        if formHz.get() != 'hnform':
            H = evalHz(fL)
            h_n = np.real(np.fft.ifft(H))
            hL = np.append(hL, h_n)
        else:
            hL = np.append(hL, lb)
            # print(hn)
        if formHz.get() != 'hnform':
            H = evalHz(fH)
            h_n = np.real(np.fft.ifft(H))
            hH = np.append(hH, h_n)
        else:
            hH = np.append(hH, lb)
            # print(hn)

        hL = applyWindow(hL)
        hH = applyWindow(hH)

        hH = -hH
        hH[N - 1 // 2] += 1

        h_n = hH + hL
        hn = np.append(hn, h_n)


# *************** extraction and processing ends here***************


def clickRec():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLING_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    for i in range(0, int(SAMPLING_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    stream.stop_stream()
    stream.close()
    p.terminate()


def clickSave():
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLING_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def clickrecANDplay():
    p = pyaudio.PyAudio()
    WIDTH = 2
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=SAMPLING_RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    for i in range(0, int(SAMPLING_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        # dat1 = np.frombuffer(data, dtype=np.float32)
        # global hn
        # numpydata = signal.fftconvolve(dat1, hn, 'full')
        # dat2 = numpydata.astype(np.float32).tostring()
        stream.write(data, CHUNK)

    print("* done")

    stream.stop_stream()
    stream.close()
    p.terminate()


def clickPlay():
    numpyframes = np.hstack(frames)
    global hn
    numpydata = signal.fftconvolve(numpyframes, hn, 'full')

    play = pyaudio.PyAudio()
    stream_play = play.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=SAMPLING_RATE,
                            output=True)
    dat = numpydata.astype(np.float32).tostring()
    stream_play.write(dat)

    stream_play.stop_stream()
    stream_play.close()
    play.terminate()


Button(app, text='Process', width=15, command=process).grid(row=9, column=1, pady=10)
bRec = Button(app, text='Record and Play', width=20, command=clickrecANDplay)
bRec.grid(row=10, column=0, padx=20, pady=10)
bPlay = Button(app, text='Play', width=20, state='disabled', command=clickPlay)
bPlay.grid(row=10, column=1, padx=20, pady=10)
bSave = Button(app, text='Save', width=20, state='disabled', command=clickSave)
bSave.grid(row=10, column=2, padx=20, pady=10)

app.mainloop()
