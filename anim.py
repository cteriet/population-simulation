import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

SZ = 7
GS = 25
INTERVAL = 200

msg = "Traffic Light Protocol: Active. Sync is now guaranteed."

def h74_e(n):
    d = [int(x) for x in f"{n:04b}"]
    p1, p2, d1, p3, d2, d3, d4 = d[0]^d[1]^d[3], d[0]^d[2]^d[3], d[0], d[1]^d[2]^d[3], d[1], d[2], d[3]
    return [p1, p2, d1, p3, d2, d3, d4]

def gdm(gs):
    m = np.ones((gs, gs), dtype=bool)

    m[:SZ, :SZ] = False; m[:SZ, -SZ:] = False
    m[-SZ:, :SZ] = False; m[-SZ:, -SZ:] = False
    m[0, :] = False; m[-1, :] = False; m[:, 0] = False; m[:, -1] = False
    return m

def aa(g):
    g[1:6, 1:6] = 1
    g[2:5, 2:5] = 0
    g[1:6, -6:-1] = 1
    g[-6:-1, 1:6] = 1
    g[-6:-1, -6:-1] = 1

    return g

def cpf(type_str, gs):
    g = np.zeros((gs, gs), dtype=int)
    g = aa(g)
    c = gs // 2

    if type_str == "START":
        g[c-2:c+3, c-2:c+3] = 1
        g[c, c] = 0

    elif type_str == "STOP":
        for i in range(5, gs-5):
            g[i, i] = 1; g[i, gs-1-i] = 1

    elif type_str == "SWITCH":
        for r in range(7, gs-7, 2):
            g[r, 5:-5] = 1

    return g

def ccf(data_str, gs=25):
    binary_data = ''.join(format(ord(c), '08b') for c in data_str)
    encoded_bits = []
    pad_len = (4 - len(binary_data) % 4) % 4
    binary_data += '0' * pad_len
    for i in range(0, len(binary_data), 4):
        encoded_bits.extend(h74_e(int(binary_data[i:i+4], 2)))

    m = gdm(gs)

    bits_per_frame = np.sum(m) - 28
    ttlf = (len(encoded_bits) + bits_per_frame - 1) // bits_per_frame

    frames = []

    for _ in range(5): frames.append(cpf("START", gs))

    for f_idx in range(ttlf):
        switch_frame = cpf("SWITCH", gs)
        for _ in range(3): frames.append(switch_frame)

        g = np.zeros((gs, gs), dtype=int)
        g = aa(g)

        header_raw = f"{f_idx+1:08b}{ttlf:08b}"
        header_bits = []
        for i in range(0, 16, 4): header_bits.extend(h74_e(int(header_raw[i:i+4], 2)))
        full_stream = header_bits + encoded_bits[f_idx*bits_per_frame : (f_idx+1)*bits_per_frame]

        fg = g.flatten(); fm = m.flatten(); stream_idx = 0
        for i in range(len(fg)):
            if fm[i]:
                if stream_idx < len(full_stream):
                    fg[i] = full_stream[stream_idx]
                    stream_idx += 1

        df = fg.reshape(gs, gs)
        for _ in range(5): frames.append(df)

    for _ in range(3): frames.append(cpf("SWITCH", gs))
    for _ in range(5): frames.append(cpf("STOP", gs))

    return frames

frames = ccf(msg, gs=GS)

fig, ax = plt.subplots(figsize=(5, 5))
mat = ax.imshow(frames[0], cmap='Greys', vmin=0, vmax=1)
plt.axis('off')
def u(f):
    mat.set_data(f)
    return [mat]
anim = animation.FuncAnimation(fig, u, frames=frames, interval=INTERVAL, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())
