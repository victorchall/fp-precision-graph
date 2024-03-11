import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import math
import struct

HIDE_FP32 = False
HIDE_BF16 = False
HIDE_FP16 = False

def on_close():
    print("Closing application")
    root.quit()
    root.destroy()

def x_in_fp16(x):
    x = np.float16(x)
    x = np.float64(x)
    return x

def x_in_fp32(x):
    x = np.float32(x)
    x = np.float64(x)
    return x

def convert_range_to_dict(range, fn):
    return_dict = {}
    for x in range:
        return_dict[x] = fn(x)
    return return_dict

def create_range(min_range:np.float64, max_range:np.float64, total_steps:int=4096):
    step_size = (max_range - min_range) / total_steps
    return np.arange(min_range, max_range, step_size)

def max_normal_exponent(exponent_bits):
    return (2 ** (exponent_bits - 1)) - 1

def get_number_parts(x):
    # Pack the float into 8 bytes (64 bits) using IEEE 754 format
    packed = struct.pack('>d', x)  # big-endian double-precision float
    bytes_ = struct.unpack('>Q', packed)[0]  # Unpack as a big-endian unsigned long 64b

    # Extract sign (bit 63), exponent (bits 62-52), and mantissa (bits 51-0)
    sign = (bytes_ >> 63) & 0x1
    exponent = ((bytes_ >> 52) & 0x7ff) - 0x3ff

    # Set the exponent bits to 0 to extract the mantissa, adjust for bias
    # Mantissa in IEEE 754 is represented with an implicit leading 1 (for normalized numbers),
    # so we restore it by adding 1 and then divide by 2^52 to get the fraction.
    mantissa = ((bytes_ & 0xfffffffffffff) | 0x10000000000000) / (2 ** 52)

    return {
        'sign': sign,
        'exponent': exponent,
        'mantissa': mantissa
    }

def exponent_bias(exponent_bits):
    # Placeholder function: calculates the bias for the given number of exponent bits ???
    return (2 ** (exponent_bits - 1)) - 1

def round_bits_to(x:np.float64, mantissa_bits:int, exponent_bits:int, clamp_to_inf:bool, flush_subnormal:bool):
    """
    Round a float to the nearest representable value with the given number of mantissa and exponent bits.
    This probably isn't 100% IEEE 754 compliant (???) but it should be close enough for visualization purposes.
    """
    if mantissa_bits > 23 or exponent_bits > 8:
        return np.float64('nan')
    if math.isnan(x):
        return np.float64('nan')

    possible_mantissas = np.power(2, mantissa_bits, dtype=np.float64)
    mantissa_max = np.float64(2.0 - 1.0 / possible_mantissas)
    max_value = np.power(2, max_normal_exponent(exponent_bits), dtype=np.float64) * mantissa_max
    if x > max_value:
        return np.float64(np.inf) if clamp_to_inf else max_value
    if x < -max_value:
        return np.float64(-np.inf) if clamp_to_inf else -max_value

    parts = get_number_parts(x)
    mantissa_rounded = math.floor(parts['mantissa'] * possible_mantissas) / possible_mantissas
    if parts['exponent'] + exponent_bias(exponent_bits) <= 0:
        if flush_subnormal:
            return -0.0 if parts['sign'] else 0.0
        else:
            while parts['exponent'] + exponent_bias(exponent_bits) <= 0:
                parts['exponent'] += 1
                mantissa_rounded = math.floor(mantissa_rounded / 2 * possible_mantissas) / possible_mantissas
                if mantissa_rounded == 0:
                    return -0.0 if parts['sign'] else 0.0
    return (-1 if parts['sign'] else 1) * math.pow(2, parts['exponent']) * mantissa_rounded

def simulate_fp_values(bits_mantissa, bits_exponent, my_range):
    k_v = {}
    for x in my_range:
        value = round_bits_to(x, bits_mantissa, bits_exponent, False, True)
        #print(f"x: {x}, value: {value}")
        k_v[x] = value
    return k_v

def simulate_int_values(bits, my_range):
    k_v = {}
    max_range = np.power(2, bits - 1, dtype=np.float64)-1
    min_range = -max_range-1

    for x in my_range:
        if x < max_range and x > min_range:
            value = int(x)
        else:
            value = max_range if x > 0 else min_range
        k_v[x] = value

    return k_v

def update_graph(toggle_fp32:bool=False, toggle_fp16:bool=False, toggle_bf16:bool=False):
    global fig, ax, canvas, HIDE_FP32, HIDE_FP16, HIDE_BF16
    HIDE_BF16 = not HIDE_BF16 if toggle_bf16 else HIDE_BF16
    HIDE_FP16 = not HIDE_FP16 if toggle_fp16 else HIDE_FP16
    HIDE_FP32 = not HIDE_FP32 if toggle_fp32 else HIDE_FP32
    
    min_range = np.float64(min_range_entry.get())
    max_range = np.float64(max_range_entry.get())
    max_window_datapoints = 4096 # defines max visual precision along real number line, "enough for a 4K monitor"
    window_range = create_range(min_range, max_range, max_window_datapoints)

    ax.clear()
    ax.set_xlim(min_range, max_range)
    ax.set_ylim(min_range, max_range)

    values_fp32 = convert_range_to_dict(window_range, x_in_fp32) # FP32: 23 mantissa bits, 8 exponent bits
    #values_fp16 = get_values_for_range(my_range, x_in_fp16, 'fp16') # FP16: 10 mantissa bits, 5 exponent bits
    values_fp16 = simulate_fp_values(10, 5, window_range) # 10 mantissa bits, 5 exponent bits
    values_bf16 = simulate_fp_values(7, 8, window_range) # 7 mantissa bits, 8 exponent bits
    values_e4m3 = simulate_fp_values(4, 3, window_range) # 4 mantissa bits, 3 exponent bits
    values_e5m2 = simulate_fp_values(5, 2, window_range) # 5 mantissa bits, 2 exponent bits
    values_int8 = simulate_int_values(8, window_range)
    # TODO: add more formats here

    # Plot everything
    xy_point_size = 0.5
    ax.scatter(list(values_fp32.keys()), list(values_fp32.values()), label='fp32', color='blue', s=xy_point_size, alpha=0.2, marker='^') if not HIDE_FP32 else None
    ax.scatter(list(values_fp16.keys()), list(values_fp16.values()), label='fp16', color='red', s=xy_point_size, alpha=0.2, marker='*') if not HIDE_FP16 else None
    ax.scatter(list(values_bf16.keys()), list(values_bf16.values()), label='bf16', color='green', s=xy_point_size, alpha=0.2, marker='d') if not HIDE_BF16 else None
    ax.scatter(list(values_e4m3.keys()), list(values_e4m3.values()), label='fp8 e4m3', color='purple', s=xy_point_size, alpha=0.2, marker='>')
    ax.scatter(list(values_e5m2.keys()), list(values_e5m2.values()), label='fp8 e5m2', color='orange', s=xy_point_size, alpha=0.2, marker='<')
    ax.scatter(list(values_int8.keys()), list(values_int8.values()), label='int8', color='black', s=xy_point_size, alpha=0.2, marker='o')

    legend_size = 10
    # hack to force the legend to display a larger marker size than the scatter
    custom_handles = [
        mlines.Line2D([], [], color='blue', marker='^', linestyle='None', markersize=legend_size, label='fp32'),
        mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=legend_size, label='fp16'),
        mlines.Line2D([], [], color='green', marker='d', linestyle='None', markersize=legend_size, label='bf16'),
        mlines.Line2D([], [], color='purple', marker='>', linestyle='None', markersize=legend_size, label='fp8 e4m2'),
        mlines.Line2D([], [], color='orange', marker='<', linestyle='None', markersize=legend_size, label='fp8 e5m2'),
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=legend_size, label='int8')
    ]

    ax.legend(handles=custom_handles, loc='upper left')

    canvas.draw()

def zoom(factor=np.e):
    global ax
    min_range = np.float64(min_range_entry.get()) # 1
    max_range = np.float64(max_range_entry.get()) # 2
    center = (min_range + max_range) / 2.0 # 1.5
    min_range = ((min_range - center) * factor) + center
    max_range = ((max_range - center) * factor) + center
    
    min_range_entry.delete(0, tk.END)
    min_range_entry.insert(0, str(min_range))
    max_range_entry.delete(0, tk.END)
    max_range_entry.insert(0, str(max_range))
    center_entry.delete(0, tk.END)
    center_entry.insert(0, str(center))

    update_graph()

def recenter(center=0.0, range=0.5):
    global ax
    center = np.float64(center)
    range = np.float64(range)

    min_range_entry.delete(0, tk.END)
    min_range_entry.insert(0, str(center-range))
    max_range_entry.delete(0, tk.END)
    max_range_entry.insert(0, str(center+range))
    center_entry.delete(0, tk.END)
    center_entry.insert(0, str(center))

    update_graph()

if __name__ == "__main__":
    root = tk.Tk()
    root.wm_title("Floating Point Visualization")


    # Main figure and axes
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    entry_frame = tk.Frame(root)
    entry_frame.pack()

    min_range_label = tk.Label(master=entry_frame, text="Min range:")
    min_range_label.grid(row=0, column=0, sticky="e")
    min_range_entry = tk.Entry(master=entry_frame, width=30)
    min_range_entry.insert(0, str(-np.e))
    min_range_entry.grid(row=0, column=1)

    max_range_label = tk.Label(master=entry_frame, text="Max range:")
    max_range_label.grid(row=1, column=0, sticky="e")
    max_range_entry = tk.Entry(master=entry_frame, width=30)
    max_range_entry.insert(0, str(np.e))
    max_range_entry.grid(row=1, column=1)

    center_label = tk.Label(master=entry_frame, text="Center:")
    center_label.grid(row=2, column=0, sticky="e")
    center_entry = tk.Entry(master=entry_frame, width=30)
    center_entry.insert(0, "0.0")
    center_entry.grid(row=2, column=1)

    update_btn = tk.Button(master=entry_frame, text="Update", command=update_graph)
    update_btn.grid(row=3, column=0, columnspan=2)

    zoom_in_btn = tk.Button(master=entry_frame, text="Zoom In", command=lambda:zoom(1/np.e))
    zoom_in_btn.grid(row=4, column=0)
    
    zoom_out_btn = tk.Button(master=entry_frame, text="Zoom Out", command=lambda:zoom(np.e))
    zoom_out_btn.grid(row=4, column=1)

    center_1e_4_btn = tk.Button(master=entry_frame, text="center on 1e-9", command=lambda:recenter(1e-9, 5e-8))
    center_1e_4_btn.grid(row=5, column=0)

    center_1e0_btn = tk.Button(master=entry_frame, text="center on 1e0", command=lambda:recenter(1e0, 5e-1))
    center_1e0_btn.grid(row=5, column=1)

    center_1e4_btn = tk.Button(master=entry_frame, text="center on 1e5", command=lambda:recenter(1e5, 5e4))
    center_1e4_btn.grid(row=6, column=0)

    center_btn = tk.Button(master=entry_frame, text="center on 0.0", command=lambda:recenter(0.0, 0.5))
    center_btn.grid(row=6, column=1)

    toggle_fp32_btn = tk.Button(master=entry_frame, text="Toggle FP32", command=lambda:update_graph(toggle_fp32=True))
    toggle_fp32_btn.grid(row=4, column=4)

    toggle_fp16_btn = tk.Button(master=entry_frame, text="Toggle FP16", command=lambda:update_graph(toggle_fp16=True))
    toggle_fp16_btn.grid(row=5, column=4)

    toggle_fp16_btn = tk.Button(master=entry_frame, text="Toggle BF16", command=lambda:update_graph(toggle_bf16=True))
    toggle_fp16_btn.grid(row=6, column=4)

    update_graph()

    root.protocol("WM_DELETE_WINDOW", on_close) # Windows OS won't shutdown without hooking the close event
    tk.mainloop()
