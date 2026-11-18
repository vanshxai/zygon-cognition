"""Pure-stdlib GGUF reader + dequantizers. No third-party libraries.

GGUF layout: header | metadata KVs | tensor infos | padding | tensor data (mmap'd).
"""
import mmap
import struct
from array import array

ORIGINAL = r"C:\Users\admin\Desktop\vansh\model\original\qwen2.5-3b-instruct-q4_k_m.original.gguf"

# ggml type id -> (name, values per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2),
}

# metadata value types
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Tensor:
    def __init__(self, name, shape, ggml_type, offset):
        self.name, self.shape, self.type_id, self.offset = name, shape, ggml_type, offset
        self.type_name, self.block_vals, self.block_bytes = GGML_TYPES[ggml_type]
        self.n_elements = 1
        for d in shape:
            self.n_elements *= d
        self.n_bytes = self.n_elements // self.block_vals * self.block_bytes

    def __repr__(self):
        return f"Tensor({self.name}, {self.shape}, {self.type_name})"


class GGUF:
    def __init__(self, path=ORIGINAL):
        self.path = path
        self._f = open(path, "rb")
        self.mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._pos = 0
        self._parse()

    # ---- primitive readers -------------------------------------------------
    def _u(self, fmt):
        size = struct.calcsize(fmt)
        v = struct.unpack_from("<" + fmt, self.mm, self._pos)[0]
        self._pos += size
        return v

    def _str(self):
        n = self._u("Q")
        s = self.mm[self._pos:self._pos + n].decode("utf-8", errors="replace")
        self._pos += n
        return s

    def _value(self, vtype):
        if vtype in _SCALAR:
            return self._u(_SCALAR[vtype])
        if vtype == 8:
            return self._str()
        if vtype == 9:
            et, n = self._u("I"), self._u("Q")
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"unknown metadata type {vtype}")

    # ---- parse -------------------------------------------------------------
    def _parse(self):
        assert self.mm[:4] == b"GGUF", "not a GGUF file"
        self._pos = 4
        self.version = self._u("I")
        n_tensors, n_kv = self._u("Q"), self._u("Q")
        self.meta = {}
        for _ in range(n_kv):
            k = self._str()
            self.meta[k] = self._value(self._u("I"))
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u("I")
            shape = tuple(self._u("Q") for _ in range(nd))  # ggml order: fastest dim first
            t, off = self._u("I"), self._u("Q")
            self.tensors[name] = Tensor(name, shape, t, off)
        align = self.meta.get("general.alignment", 32)
        self.data_start = (self._pos + align - 1) // align * align

    # ---- tensor access -----------------------------------------------------
    def raw(self, name):
        t = self.tensors[name]
        s = self.data_start + t.offset
        return self.mm[s:s + t.n_bytes]

    def dequantize(self, name):
        """Whole tensor -> array('f') of floats, row-major with shape reversed
        (so a ggml shape (in, out) is a matrix of `out` rows, each `in` long)."""
        t = self.tensors[name]
        return DEQUANT[t.type_name](self.raw(name), t.n_elements)

    def row(self, name, r):
        """Dequantize a single row (cheap: only touches that row's blocks)."""
        t = self.tensors[name]
        row_len = t.shape[0]
        bpr = row_len // t.block_vals * t.block_bytes
        s = self.data_start + t.offset + r * bpr
        return DEQUANT[t.type_name](self.mm[s:s + bpr], row_len)


# ---- dequantizers ----------------------------------------------------------
def _f16(b, o):
    return struct.unpack_from("<e", b, o)[0]


def _deq_f32(b, n):
    a = array("f"); a.frombytes(bytes(b)); return a


def _deq_f16(b, n):
    return array("f", struct.unpack(f"<{n}e", b))


def _deq_bf16(b, n):
    u = struct.unpack(f"<{n}H", b)
    return array("f", struct.unpack(f"<{n}f", b"".join(struct.pack("<I", x << 16) for x in u)))


def _scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def _deq_q4_k(b, n):
    out = array("f")
    for bo in range(0, len(b), 144):
        d, dmin = _f16(b, bo), _f16(b, bo + 2)
        sc = b[bo + 4:bo + 16]
        qs = b[bo + 16:bo + 144]
        for g in range(4):  # 4 groups of 64 values = 2 sub-blocks of 32 sharing the same 32 bytes
            s1, m1 = _scale_min_k4(2 * g, sc)
            s2, m2 = _scale_min_k4(2 * g + 1, sc)
            q = qs[32 * g:32 * g + 32]
            d1, mm1, d2, mm2 = d * s1, dmin * m1, d * s2, dmin * m2
            out.extend(d1 * (x & 0xF) - mm1 for x in q)
            out.extend(d2 * (x >> 4) - mm2 for x in q)
    return out


def _deq_q6_k(b, n):
    out = array("f")
    for bo in range(0, len(b), 210):
        ql, qh = b[bo:bo + 128], b[bo + 128:bo + 192]
        sc = struct.unpack_from("<16b", b, bo + 192)
        d = _f16(b, bo + 208)
        y = [0.0] * 256
        for half in range(2):
            qlo, qho, so, yo = 64 * half, 32 * half, 8 * half, 128 * half
            for l in range(32):
                i = l // 16
                h = qh[qho + l]
                a, c = ql[qlo + l], ql[qlo + l + 32]
                y[yo + l] = d * sc[so + i] * (((a & 0xF) | ((h & 3) << 4)) - 32)
                y[yo + l + 32] = d * sc[so + i + 2] * (((c & 0xF) | (((h >> 2) & 3) << 4)) - 32)
                y[yo + l + 64] = d * sc[so + i + 4] * (((a >> 4) | (((h >> 4) & 3) << 4)) - 32)
                y[yo + l + 96] = d * sc[so + i + 6] * (((c >> 4) | (((h >> 6) & 3) << 4)) - 32)
        out.extend(y)
    return out


def _deq_q8_0(b, n):
    out = array("f")
    for bo in range(0, len(b), 34):
        d = _f16(b, bo)
        out.extend(d * x for x in struct.unpack_from("<32b", b, bo + 2))
    return out


DEQUANT = {"F32": _deq_f32, "F16": _deq_f16, "BF16": _deq_bf16,
           "Q4_K": _deq_q4_k, "Q6_K": _deq_q6_k, "Q8_0": _deq_q8_0}


# ---- vectorised numpy dequantizers (fast path; the loops above are the readable reference) ----
import numpy as np


def np_q4_k(raw, n):
    b = np.frombuffer(raw, np.uint8).reshape(-1, 144)
    N = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)      # (N,1)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    s = b[:, 4:16]
    sc = np.empty((N, 8), np.float32); mn = np.empty((N, 8), np.float32)
    sc[:, :4] = s[:, :4] & 63; mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 15) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    qs = b[:, 16:].reshape(N, 4, 32)
    q = np.stack([qs & 15, qs >> 4], axis=2).astype(np.float32)    # (N,4,2,32)
    sc = (d * sc).reshape(N, 4, 2, 1)
    mn = (dmin * mn).reshape(N, 4, 2, 1)
    return (sc * q - mn).reshape(-1)


def np_q6_k(raw, n):
    b = np.frombuffer(raw, np.uint8).reshape(-1, 210)
    N = b.shape[0]
    ql = b[:, 0:128].reshape(N, 2, 64)
    qh = b[:, 128:192].reshape(N, 2, 32)
    sc = b[:, 192:208].copy().view(np.int8).reshape(N, 2, 4, 2).astype(np.float32)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(N, 1, 1, 1)
    a, c = ql[:, :, :32], ql[:, :, 32:]
    q = np.stack([(a & 15) | ((qh & 3) << 4), (c & 15) | (((qh >> 2) & 3) << 4),
                  (a >> 4) | (((qh >> 4) & 3) << 4), (c >> 4) | (((qh >> 6) & 3) << 4)], axis=2)
    q = q.astype(np.float32) - 32                                  # (N,2,4,32)
    return (d * np.repeat(sc, 16, axis=3) * q).reshape(-1)


def np_q8_0(raw, n):
    b = np.frombuffer(raw, np.uint8).reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (d * b[:, 2:].copy().view(np.int8).astype(np.float32)).reshape(-1)


NP_DEQUANT = {
    "F32": lambda r, n: np.frombuffer(r, np.float32),
    "F16": lambda r, n: np.frombuffer(r, np.float16).astype(np.float32),
    "Q4_K": np_q4_k, "Q6_K": np_q6_k, "Q8_0": np_q8_0,
}


def np_tensor(g, name):
    """Dequantized tensor as float32 ndarray in numpy shape (out, in)."""
    t = g.tensors[name]
    return NP_DEQUANT[t.type_name](g.raw(name), t.n_elements).reshape(t.shape[::-1])
