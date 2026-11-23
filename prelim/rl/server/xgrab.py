# Grab the root window of an X display to PNG (stdlib + libX11 via ctypes).  python3 xgrab.py :1 out.png
import ctypes, struct, sys, zlib
X = ctypes.cdll.LoadLibrary("libX11.so.6")
X.XOpenDisplay.restype = ctypes.c_void_p
X.XDefaultRootWindow.restype = ctypes.c_ulong
X.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
X.XGetImage.restype = ctypes.c_void_p
X.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
class XImage(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int), ("format", ctypes.c_int),
                ("data", ctypes.POINTER(ctypes.c_ubyte)), ("byte_order", ctypes.c_int), ("bitmap_unit", ctypes.c_int),
                ("bitmap_bit_order", ctypes.c_int), ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int),
                ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int)]
d = X.XOpenDisplay(sys.argv[1].encode())
root = X.XDefaultRootWindow(d)
class Attr(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("width", ctypes.c_int), ("height", ctypes.c_int)] + [("pad", ctypes.c_byte * 256)]
a = Attr(); X.XGetWindowAttributes(ctypes.c_void_p(d), ctypes.c_ulong(root), ctypes.byref(a))
img = ctypes.cast(X.XGetImage(d, root, 0, 0, a.width, a.height, 0xFFFFFFFF, 2), ctypes.POINTER(XImage)).contents
raw = ctypes.string_at(img.data, img.bytes_per_line * img.height)
rows = b"".join(b"\0" + bytes(c for px in range(img.width) for c in (raw[y*img.bytes_per_line+4*px+2], raw[y*img.bytes_per_line+4*px+1], raw[y*img.bytes_per_line+4*px])) for y in range(img.height))
chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
open(sys.argv[2], "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", img.width, img.height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))
print(img.width, img.height)
