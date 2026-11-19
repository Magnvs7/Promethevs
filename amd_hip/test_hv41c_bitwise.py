import ctypes
import os
import re
import sys
import time
import numpy as np


SHAPES = [
    ("tiny_edge", 128, 256),
    ("proj_512x5120", 512, 5120),
    ("shared_gate_up", 1280, 5120),
    ("attn_q_wkv_a", 2304, 5120),
    ("attn_output", 5120, 2304),
    ("shared_down", 5120, 1280),
    ("attn_q_b", 8192, 5120),
    ("lm_head_chunk", 32768, 1280),
]
BATCH_ROWS = [1, 2, 5, 8, 16, 64]


def build_decode_lut() -> np.ndarray:
    lut = np.empty(256, dtype=np.int64)
    for b in range(256):
        sign = (b >> 7) & 1
        exp = (b >> 3) & 0xF
        mant = b & 7
        mag = mant if exp == 0 else ((8 + mant) << (exp - 1))
        lut[b] = -mag if sign else mag
    return lut


DECODE_LUT_I64 = build_decode_lut()
DECODE_LUT_F64 = DECODE_LUT_I64.astype(np.float64)


def round_bf16_in_f32_u32(arr_f32: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(arr_f32, dtype=np.float32).view(np.uint32)
    return (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)


def cpu_ref_hv41f_exact_i64(
    codes_u8: np.ndarray, scales_i8: np.ndarray, a_i32: np.ndarray, s_f32: np.ndarray
) -> np.ndarray:
    """Exact int64 inner-product reference (safe for arbitrary full-range int32 edge inputs)."""
    N, K = codes_u8.shape
    rows = a_i32.shape[0]
    nkb = K >> 5
    w_blocks = DECODE_LUT_I64[codes_u8].reshape(N, nkb, 32)
    a_blocks = a_i32.astype(np.int64).reshape(rows, nkb, 32)
    sc_full = np.repeat(scales_i8, 32, axis=0)  # (N, nkb)

    acc = np.zeros((rows, N), dtype=np.float32)
    inv_2_18 = np.float64(2.0 ** -18)
    for kb in range(nkb):
        dot = (a_blocks[:, kb, :][:, None, :] * w_blocks[:, kb, :][None, :, :]).sum(
            axis=2, dtype=np.int64
        )
        scale_e = np.ldexp(inv_2_18, sc_full[:, kb].astype(np.int32))[None, :]
        bval = (dot.astype(np.float64) * scale_e).astype(np.float32)
        term = bval * s_f32[:, kb : kb + 1]
        acc = term + acc
    return round_bf16_in_f32_u32(acc)


def cpu_ref_hv41f_fast(
    codes_u8: np.ndarray, scales_i8: np.ndarray, a_i32: np.ndarray, s_f32: np.ndarray
) -> np.ndarray:
    """High-speed BLAS float64 reference (100% integer-exact when sum |w*a| <= 2^53)."""
    N, K = codes_u8.shape
    rows = a_i32.shape[0]
    nkb = K >> 5
    w_blocks = DECODE_LUT_F64[codes_u8].reshape(N, nkb, 32)
    a_blocks = a_i32.astype(np.float64).reshape(rows, nkb, 32)
    sc_full = np.repeat(scales_i8, 32, axis=0)  # (N, nkb)

    acc = np.zeros((rows, N), dtype=np.float32)
    inv_2_18 = np.float64(2.0 ** -18)
    for kb in range(nkb):
        dot_f64 = a_blocks[:, kb, :] @ w_blocks[:, kb, :].T
        scale_e = np.ldexp(inv_2_18, sc_full[:, kb].astype(np.int32))[None, :]
        bval = (dot_f64 * scale_e).astype(np.float32)
        term = bval * s_f32[:, kb : kb + 1]
        acc = term + acc
    return round_bf16_in_f32_u32(acc)


class DirectHipHv41c:
    """Sanctioned fallback using WHQL-signed amd_comgr_2.dll + amdhip64_6.dll when SAC blocks loading rebuilt DLLs."""

    AMD_COMGR_LANGUAGE_HIP = 3
    AMD_COMGR_DATA_KIND_SOURCE = 1
    AMD_COMGR_DATA_KIND_DIAGNOSTIC = 4
    AMD_COMGR_DATA_KIND_LOG = 5
    AMD_COMGR_DATA_KIND_EXECUTABLE = 8
    AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC = 2
    AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = 4
    AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 7

    HIP_MEMCPY_HOST_TO_DEVICE = 1
    HIP_MEMCPY_DEVICE_TO_HOST = 2
    HIP_DEVICE_ATTR_MP_COUNT = 63

    def __init__(self, cpp_path: str):
        self.cpp_path = cpp_path
        self.backend_name = f"DirectHipHv41c(amd_comgr_2.dll + amdhip64_6.dll <- {os.path.basename(cpp_path)})"
        self._last_error = b""
        self._initialized = False
        self._init_info = b""
        self._handles = {}
        self._next_handle = 1
        self._resident_bytes = 0
        self._d_a = None
        self._d_a_bytes = 0
        self._d_s = None
        self._d_s_bytes = 0
        self._d_out = None
        self._d_out_bytes = 0
        self._hsaco_buf = None
        self.comgr = None
        self.hip = None
        self.module = ctypes.c_void_p()
        self.fn_m1 = ctypes.c_void_p()
        self.fn_tile4 = ctypes.c_void_p()
        self.fn_bf16_seq = ctypes.c_void_p()
        self.fn_bf16_w32 = ctypes.c_void_p()
        self.fn_sattn = ctypes.c_void_p()
        self.fn_plane_restore = ctypes.c_void_p()
        self.fn_fp4_linear = ctypes.c_void_p()
        self._bf16_handles = {}
        self._d_bf16_x = None
        self._d_bf16_x_bytes = 0
        self._d_bf16_y = None
        self._d_bf16_y_bytes = 0
        self._d_exptable = None
        self._exptable_count = 0
        self.stream = ctypes.c_void_p()
        self.ev_start = ctypes.c_void_p()
        self.ev_stop = ctypes.c_void_p()

    def _set_err(self, msg: str):
        self._last_error = msg.encode("utf-8", errors="replace")

    def hv41c_error(self) -> bytes:
        return self._last_error

    def _bind_comgr(self):
        c = ctypes.WinDLL("amd_comgr_2.dll")
        c.amd_comgr_status_string.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        c.amd_comgr_status_string.restype = ctypes.c_int
        c.amd_comgr_create_data.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
        c.amd_comgr_create_data.restype = ctypes.c_int
        c.amd_comgr_release_data.argtypes = [ctypes.c_uint64]
        c.amd_comgr_release_data.restype = ctypes.c_int
        c.amd_comgr_set_data.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_char_p]
        c.amd_comgr_set_data.restype = ctypes.c_int
        c.amd_comgr_set_data_name.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        c.amd_comgr_set_data_name.restype = ctypes.c_int
        c.amd_comgr_get_data.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
        c.amd_comgr_get_data.restype = ctypes.c_int
        c.amd_comgr_create_data_set.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        c.amd_comgr_create_data_set.restype = ctypes.c_int
        c.amd_comgr_destroy_data_set.argtypes = [ctypes.c_uint64]
        c.amd_comgr_destroy_data_set.restype = ctypes.c_int
        c.amd_comgr_data_set_add.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        c.amd_comgr_data_set_add.restype = ctypes.c_int
        c.amd_comgr_action_data_count.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
        c.amd_comgr_action_data_count.restype = ctypes.c_int
        c.amd_comgr_action_data_get_data.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
        c.amd_comgr_action_data_get_data.restype = ctypes.c_int
        c.amd_comgr_create_action_info.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        c.amd_comgr_create_action_info.restype = ctypes.c_int
        c.amd_comgr_destroy_action_info.argtypes = [ctypes.c_uint64]
        c.amd_comgr_destroy_action_info.restype = ctypes.c_int
        c.amd_comgr_action_info_set_isa_name.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        c.amd_comgr_action_info_set_isa_name.restype = ctypes.c_int
        c.amd_comgr_action_info_set_language.argtypes = [ctypes.c_uint64, ctypes.c_int]
        c.amd_comgr_action_info_set_language.restype = ctypes.c_int
        c.amd_comgr_action_info_set_logging.argtypes = [ctypes.c_uint64, ctypes.c_bool]
        c.amd_comgr_action_info_set_logging.restype = ctypes.c_int
        c.amd_comgr_action_info_set_option_list.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
        c.amd_comgr_action_info_set_option_list.restype = ctypes.c_int
        c.amd_comgr_do_action.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        c.amd_comgr_do_action.restype = ctypes.c_int
        self.comgr = c

    def _bind_hip(self):
        h = ctypes.WinDLL("amdhip64_6.dll")
        h.hipGetErrorString.argtypes = [ctypes.c_int]
        h.hipGetErrorString.restype = ctypes.c_char_p
        h.hipInit.argtypes = [ctypes.c_uint]
        h.hipInit.restype = ctypes.c_int
        h.hipSetDevice.argtypes = [ctypes.c_int]
        h.hipSetDevice.restype = ctypes.c_int
        h.hipDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        h.hipDeviceGetName.restype = ctypes.c_int
        h.hipDeviceComputeCapability.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        h.hipDeviceComputeCapability.restype = ctypes.c_int
        h.hipDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        h.hipDeviceGetAttribute.restype = ctypes.c_int
        h.hipMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        h.hipMemGetInfo.restype = ctypes.c_int
        h.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        h.hipModuleLoadData.restype = ctypes.c_int
        h.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        h.hipModuleGetFunction.restype = ctypes.c_int
        h.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        h.hipMalloc.restype = ctypes.c_int
        h.hipFree.argtypes = [ctypes.c_void_p]
        h.hipFree.restype = ctypes.c_int
        h.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        h.hipMemcpy.restype = ctypes.c_int
        h.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        h.hipMemcpyAsync.restype = ctypes.c_int
        h.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        h.hipStreamCreate.restype = ctypes.c_int
        h.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        h.hipStreamSynchronize.restype = ctypes.c_int
        h.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        h.hipEventCreate.restype = ctypes.c_int
        h.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        h.hipEventRecord.restype = ctypes.c_int
        h.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        h.hipEventElapsedTime.restype = ctypes.c_int
        h.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ]
        h.hipModuleLaunchKernel.restype = ctypes.c_int
        self.hip = h

    def _comgr_logs(self, ds_handle: int) -> str:
        c = self.comgr
        parts = []
        for kind in (self.AMD_COMGR_DATA_KIND_LOG, self.AMD_COMGR_DATA_KIND_DIAGNOSTIC):
            cnt = ctypes.c_size_t(0)
            if c.amd_comgr_action_data_count(ds_handle, kind, ctypes.byref(cnt)) != 0:
                continue
            for i in range(cnt.value):
                d = ctypes.c_uint64(0)
                if c.amd_comgr_action_data_get_data(ds_handle, kind, i, ctypes.byref(d)) != 0:
                    continue
                sz = ctypes.c_size_t(0)
                if c.amd_comgr_get_data(d.value, ctypes.byref(sz), None) == 0 and sz.value > 0:
                    buf = ctypes.create_string_buffer(sz.value)
                    if c.amd_comgr_get_data(d.value, ctypes.byref(sz), buf) == 0:
                        parts.append(buf.raw[: sz.value].decode("utf-8", errors="replace").rstrip("\x00"))
                c.amd_comgr_release_data(d.value)
        return "\n".join(parts)

    def _compile_hsaco(self, isa_name: bytes) -> bytes:
        with open(self.cpp_path, "r", encoding="utf-8") as f:
            cpp_text = f.read()
        parts = re.findall(r'kHipKernelSource\d*\s*=\s*R"HIP\((.*?)\)HIP";', cpp_text, re.DOTALL)
        if not parts:
            raise RuntimeError(f"Could not extract kHipKernelSource from {self.cpp_path}")
        hip_src = "".join(parts).encode("utf-8")

        c = self.comgr
        src_data = ctypes.c_uint64(0)
        src_set = ctypes.c_uint64(0)
        bc_set = ctypes.c_uint64(0)
        reloc_set = ctypes.c_uint64(0)
        exec_set = ctypes.c_uint64(0)
        info = ctypes.c_uint64(0)

        def ck(rc: int, step: str, ds_h: int = 0):
            if rc != 0:
                sp = ctypes.c_char_p()
                c.amd_comgr_status_string(rc, ctypes.byref(sp))
                s = sp.value.decode("utf-8", errors="replace") if sp.value else "?"
                log = self._comgr_logs(ds_h) if ds_h else ""
                raise RuntimeError(f"comgr {step} failed ({rc}: {s})" + (f" | {log}" if log else ""))

        try:
            ck(c.amd_comgr_create_data(self.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(src_data)), "create_data")
            ck(c.amd_comgr_set_data(src_data.value, len(hip_src), hip_src), "set_data")
            ck(c.amd_comgr_set_data_name(src_data.value, b"hv41_fp8_linear.hip"), "set_data_name")

            ck(c.amd_comgr_create_data_set(ctypes.byref(src_set)), "create_data_set(src)")
            ck(c.amd_comgr_data_set_add(src_set.value, src_data.value), "data_set_add")

            ck(c.amd_comgr_create_action_info(ctypes.byref(info)), "create_action_info")
            ck(c.amd_comgr_action_info_set_isa_name(info.value, isa_name), "set_isa_name")
            ck(c.amd_comgr_action_info_set_language(info.value, self.AMD_COMGR_LANGUAGE_HIP), "set_language")
            c.amd_comgr_action_info_set_logging(info.value, True)
            opts = (ctypes.c_char_p * 3)(b"-O3", b"-mcumode", b"-ffp-contract=off")
            ck(c.amd_comgr_action_info_set_option_list(info.value, opts, 3), "set_option_list")

            ck(c.amd_comgr_create_data_set(ctypes.byref(bc_set)), "create_data_set(bc)")
            ck(c.amd_comgr_do_action(self.AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info.value, src_set.value, bc_set.value), "COMPILE_SOURCE_TO_BC", bc_set.value)

            c.amd_comgr_destroy_action_info(info.value)
            info = ctypes.c_uint64(0)
            ck(c.amd_comgr_create_action_info(ctypes.byref(info)), "reinit_action_info")
            ck(c.amd_comgr_action_info_set_isa_name(info.value, isa_name), "set_isa_name(codegen)")
            c.amd_comgr_action_info_set_logging(info.value, True)
            cg_opts = (ctypes.c_char_p * 2)(b"-O3", b"-mcumode")
            ck(c.amd_comgr_action_info_set_option_list(info.value, cg_opts, 2), "set_option_list(codegen)")

            ck(c.amd_comgr_create_data_set(ctypes.byref(reloc_set)), "create_data_set(reloc)")
            ck(c.amd_comgr_do_action(self.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info.value, bc_set.value, reloc_set.value), "CODEGEN_BC_TO_RELOCATABLE", reloc_set.value)

            ck(c.amd_comgr_create_data_set(ctypes.byref(exec_set)), "create_data_set(exec)")
            ck(c.amd_comgr_do_action(self.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info.value, reloc_set.value, exec_set.value), "LINK_RELOCATABLE_TO_EXECUTABLE", exec_set.value)

            cnt = ctypes.c_size_t(0)
            ck(c.amd_comgr_action_data_count(exec_set.value, self.AMD_COMGR_DATA_KIND_EXECUTABLE, ctypes.byref(cnt)), "action_data_count")
            if cnt.value == 0:
                raise RuntimeError("comgr produced 0 executables")
            exe_obj = ctypes.c_uint64(0)
            ck(c.amd_comgr_action_data_get_data(exec_set.value, self.AMD_COMGR_DATA_KIND_EXECUTABLE, 0, ctypes.byref(exe_obj)), "get_exe")
            try:
                sz = ctypes.c_size_t(0)
                ck(c.amd_comgr_get_data(exe_obj.value, ctypes.byref(sz), None), "get_data(size)")
                buf = ctypes.create_string_buffer(sz.value)
                ck(c.amd_comgr_get_data(exe_obj.value, ctypes.byref(sz), buf), "get_data(bytes)")
                return buf.raw[: sz.value]
            finally:
                c.amd_comgr_release_data(exe_obj.value)
        finally:
            for ds in (exec_set, reloc_set, bc_set, src_set):
                if ds.value:
                    c.amd_comgr_destroy_data_set(ds.value)
            if info.value:
                c.amd_comgr_destroy_action_info(info.value)
            if src_data.value:
                c.amd_comgr_release_data(src_data.value)

    def _ensure_scratch(self, need_a: int, need_s: int, need_out: int):
        h = self.hip
        for attr_p, attr_sz, need, name in (
            ("_d_a", "_d_a_bytes", need_a, "d_a"),
            ("_d_s", "_d_s_bytes", need_s, "d_s"),
            ("_d_out", "_d_out_bytes", need_out, "d_out"),
        ):
            cur_sz = getattr(self, attr_sz)
            cur_p = getattr(self, attr_p)
            if cur_sz >= need and cur_p:
                continue
            if cur_p:
                h.hipFree(ctypes.c_void_p(cur_p))
                setattr(self, attr_p, None)
                setattr(self, attr_sz, 0)
            p = ctypes.c_void_p()
            rc = h.hipMalloc(ctypes.byref(p), need)
            if rc != 0 or not p.value:
                raise RuntimeError(f"hipMalloc({name}, {need}) failed: {rc}")
            setattr(self, attr_p, p.value)
            setattr(self, attr_sz, need)

    def hv41c_init(self, info_buf, info_buf_size: int) -> int:
        try:
            self._bind_comgr()
            self._bind_hip()
            h = self.hip
            for fn_name, rc in (
                ("hipInit", h.hipInit(0)),
                ("hipSetDevice", h.hipSetDevice(0)),
            ):
                if rc != 0:
                    self._set_err(f"{fn_name} failed ({rc}: {h.hipGetErrorString(rc).decode()})")
                    return -1

            name_buf = ctypes.create_string_buffer(256)
            if h.hipDeviceGetName(name_buf, 256, 0) != 0:
                name_buf.value = b"AMD Radeon GPU"
            maj, min_ = ctypes.c_int(10), ctypes.c_int(3)
            h.hipDeviceComputeCapability(ctypes.byref(maj), ctypes.byref(min_), 0)
            wgps = ctypes.c_int(0)
            h.hipDeviceGetAttribute(ctypes.byref(wgps), self.HIP_DEVICE_ATTR_MP_COUNT, 0)
            arch = b"gfx1031" if (maj.value == 10 and min_.value == 3) else f"gfx{maj.value}{min_.value}0".encode()
            isa = b"amdgcn-amd-amdhsa--" + arch

            hsaco = self._compile_hsaco(isa)
            self._hsaco_buf = ctypes.create_string_buffer(hsaco, len(hsaco))
            rc = h.hipModuleLoadData(ctypes.byref(self.module), self._hsaco_buf)
            if rc != 0:
                self._set_err(f"hipModuleLoadData failed ({rc})")
                return -1
            rc = h.hipModuleGetFunction(ctypes.byref(self.fn_m1), self.module, b"hv41_fp8_linear_m1")
            if rc != 0:
                self._set_err(f"hipModuleGetFunction(m1) failed ({rc})")
                return -1
            rc = h.hipModuleGetFunction(ctypes.byref(self.fn_tile4), self.module, b"hv41_fp8_linear_tile4")
            if rc != 0:
                self._set_err(f"hipModuleGetFunction(tile4) failed ({rc})")
                return -1
            for attr, kname in (
                ("fn_bf16_seq", b"hv41_bf16_seq_kernel"),
                ("fn_bf16_w32", b"hv41_bf16_w32_kernel"),
                ("fn_sattn", b"hv41_sattn_kernel"),
                ("fn_plane_restore", b"hv41_plane_restore_kernel"),
                ("fn_fp4_linear", b"hv41_fp4_linear_kernel"),
            ):
                rc = h.hipModuleGetFunction(ctypes.byref(getattr(self, attr)), self.module, kname)
                if rc != 0:
                    self._set_err(f"hipModuleGetFunction({kname.decode()}) failed ({rc})")
                    return -1
            if h.hipStreamCreate(ctypes.byref(self.stream)) != 0:
                self._set_err("hipStreamCreate failed")
                return -1
            if h.hipEventCreate(ctypes.byref(self.ev_start)) != 0 or h.hipEventCreate(ctypes.byref(self.ev_stop)) != 0:
                self._set_err("hipEventCreate failed")
                return -1

            self._ensure_scratch(64 * 8192 * 4, 64 * 256 * 4, 64 * 32768 * 4)
            vfree, vtotal = ctypes.c_size_t(0), ctypes.c_size_t(0)
            h.hipMemGetInfo(ctypes.byref(vfree), ctypes.byref(vtotal))
            info_str = (
                f"{name_buf.value.decode()} ({arch.decode()} HIP/comgr-direct), "
                f"{wgps.value} WGPs, VRAM free {vfree.value >> 20} of {vtotal.value >> 20} MiB at init"
            )
            self._init_info = info_str.encode("utf-8")
            if info_buf is not None and info_buf_size > 0:
                trimmed = self._init_info[: info_buf_size - 1]
                ctypes.memmove(info_buf, trimmed + b"\x00", len(trimmed) + 1)
            self._initialized = True
            return 0
        except Exception as exc:
            self._set_err(str(exc))
            return -1

    def hv41c_upload(self, codes_ptr: int, scales_ptr: int, N: int, K: int):
        try:
            h = self.hip
            codes_arr = np.ctypeslib.as_array((ctypes.c_uint8 * (N * K)).from_address(codes_ptr)).reshape(N, K)
            packed = np.ascontiguousarray(codes_arr.reshape(N // 32, 32, K // 32, 8, 4).transpose(0, 2, 3, 1, 4))
            codes_bytes = N * K
            scales_bytes = (N // 32) * (K // 32)

            d_codes = ctypes.c_void_p()
            d_scales = ctypes.c_void_p()
            if h.hipMalloc(ctypes.byref(d_codes), codes_bytes) != 0:
                self._set_err("hipMalloc d_codes failed")
                return None
            if h.hipMalloc(ctypes.byref(d_scales), scales_bytes) != 0:
                h.hipFree(d_codes)
                self._set_err("hipMalloc d_scales failed")
                return None
            if h.hipMemcpy(d_codes, ctypes.c_void_p(packed.ctypes.data), codes_bytes, self.HIP_MEMCPY_HOST_TO_DEVICE) != 0:
                h.hipFree(d_codes)
                h.hipFree(d_scales)
                self._set_err("hipMemcpy d_codes failed")
                return None
            if h.hipMemcpy(d_scales, ctypes.c_void_p(scales_ptr), scales_bytes, self.HIP_MEMCPY_HOST_TO_DEVICE) != 0:
                h.hipFree(d_codes)
                h.hipFree(d_scales)
                self._set_err("hipMemcpy d_scales failed")
                return None

            hid = self._next_handle
            self._next_handle += 1
            self._handles[hid] = {"d_codes": d_codes.value, "d_scales": d_scales.value, "N": N, "K": K}
            self._resident_bytes += codes_bytes + scales_bytes
            return hid
        except Exception as exc:
            self._set_err(str(exc))
            return None

    def reset_resident(self):
        for w in self._handles.values():
            self.hip.hipFree(ctypes.c_void_p(w["d_codes"]))
            self.hip.hipFree(ctypes.c_void_p(w["d_scales"]))
        self._handles.clear()
        self._resident_bytes = 0

    def hv41c_linear(self, handle: int, a_ptr: int, s_ptr: int, rows: int, out_ptr: int, out_ms_ptr) -> int:
        try:
            w = self._handles.get(handle)
            if not w:
                self._set_err("invalid handle")
                return -2
            h = self.hip
            N, K = w["N"], w["K"]
            nkb = K >> 5
            a_bytes = rows * K * 4
            s_bytes = rows * nkb * 4
            out_bytes = rows * N * 4
            self._ensure_scratch(a_bytes, s_bytes, out_bytes)

            if h.hipMemcpyAsync(ctypes.c_void_p(self._d_a), ctypes.c_void_p(a_ptr), a_bytes, self.HIP_MEMCPY_HOST_TO_DEVICE, self.stream) != 0:
                return -4
            if h.hipMemcpyAsync(ctypes.c_void_p(self._d_s), ctypes.c_void_p(s_ptr), s_bytes, self.HIP_MEMCPY_HOST_TO_DEVICE, self.stream) != 0:
                return -4
            if out_ms_ptr is not None:
                h.hipEventRecord(self.ev_start, self.stream)

            p_codes = ctypes.c_void_p(w["d_codes"])
            p_scales = ctypes.c_void_p(w["d_scales"])
            p_a = ctypes.c_void_p(self._d_a)
            p_s = ctypes.c_void_p(self._d_s)
            p_out = ctypes.c_void_p(self._d_out)
            p_N = ctypes.c_int(N)
            p_K = ctypes.c_int(K)
            grid_x = N >> 5

            if rows == 1:
                args = (ctypes.c_void_p * 7)(
                    ctypes.addressof(p_codes), ctypes.addressof(p_scales),
                    ctypes.addressof(p_a), ctypes.addressof(p_s),
                    ctypes.addressof(p_out), ctypes.addressof(p_N), ctypes.addressof(p_K),
                )
                rc = h.hipModuleLaunchKernel(self.fn_m1, grid_x, 1, 1, 32, 1, 1, 0, self.stream, args, None)
            else:
                p_rows = ctypes.c_int(rows)
                grid_y = (rows + 3) // 4
                args = (ctypes.c_void_p * 8)(
                    ctypes.addressof(p_codes), ctypes.addressof(p_scales),
                    ctypes.addressof(p_a), ctypes.addressof(p_s),
                    ctypes.addressof(p_out), ctypes.addressof(p_N), ctypes.addressof(p_K), ctypes.addressof(p_rows),
                )
                rc = h.hipModuleLaunchKernel(self.fn_tile4, grid_x, grid_y, 1, 32, 1, 1, 0, self.stream, args, None)
            if rc != 0:
                self._set_err(f"hipModuleLaunchKernel failed ({rc})")
                return -4

            if out_ms_ptr is not None:
                h.hipEventRecord(self.ev_stop, self.stream)
            if h.hipMemcpyAsync(ctypes.c_void_p(out_ptr), ctypes.c_void_p(self._d_out), out_bytes, self.HIP_MEMCPY_DEVICE_TO_HOST, self.stream) != 0:
                return -4
            if h.hipStreamSynchronize(self.stream) != 0:
                return -4
            if out_ms_ptr is not None:
                ms = ctypes.c_float(0.0)
                if h.hipEventElapsedTime(ctypes.byref(ms), self.ev_start, self.ev_stop) == 0:
                    ctypes.cast(out_ms_ptr, ctypes.POINTER(ctypes.c_float))[0] = ms.value
                else:
                    ctypes.cast(out_ms_ptr, ctypes.POINTER(ctypes.c_float))[0] = 0.0
            return 0
        except Exception as exc:
            self._set_err(str(exc))
            return -4

    def hv41c_upload_bf16(self, w_ptr: int, N: int, K: int):
        h = self.hip
        nbytes = N * K * 2
        d_w = ctypes.c_void_p()
        if h.hipMalloc(ctypes.byref(d_w), nbytes) != 0:
            return None
        if h.hipMemcpy(d_w, ctypes.c_void_p(w_ptr), nbytes, self.HIP_MEMCPY_HOST_TO_DEVICE) != 0:
            h.hipFree(d_w)
            return None
        hid = self._next_handle
        self._next_handle += 1
        self._bf16_handles[hid] = {"d_W": d_w.value, "N": N, "K": K}
        self._resident_bytes += nbytes
        return hid

    def _run_bf16(self, fn, handle: int, x_ptr: int, M: int, ng: int, K_x: int, bf16_round: int, y_ptr: int, is_seq: bool) -> int:
        w = self._bf16_handles.get(handle)
        if not w:
            return -2
        h = self.hip
        N, K = w["N"], w["K"]
        x_bytes = M * K_x * 4
        y_bytes = M * N * 4
        if self._d_bf16_x_bytes < x_bytes:
            if self._d_bf16_x:
                h.hipFree(ctypes.c_void_p(self._d_bf16_x))
            p = ctypes.c_void_p()
            assert h.hipMalloc(ctypes.byref(p), x_bytes) == 0
            self._d_bf16_x = p.value
            self._d_bf16_x_bytes = x_bytes
        if self._d_bf16_y_bytes < y_bytes:
            if self._d_bf16_y:
                h.hipFree(ctypes.c_void_p(self._d_bf16_y))
            p = ctypes.c_void_p()
            assert h.hipMalloc(ctypes.byref(p), y_bytes) == 0
            self._d_bf16_y = p.value
            self._d_bf16_y_bytes = y_bytes

        assert h.hipMemcpyAsync(ctypes.c_void_p(self._d_bf16_x), ctypes.c_void_p(x_ptr), x_bytes, self.HIP_MEMCPY_HOST_TO_DEVICE, self.stream) == 0
        p_W = ctypes.c_void_p(w["d_W"])
        p_x = ctypes.c_void_p(self._d_bf16_x)
        p_y = ctypes.c_void_p(self._d_bf16_y)
        p_M = ctypes.c_int(M)
        p_N = ctypes.c_int(N)
        p_K = ctypes.c_int(K)
        p_ng = ctypes.c_int(ng)
        p_Kx = ctypes.c_int(K_x)
        p_rnd = ctypes.c_int(1 if bf16_round else 0)
        args = (ctypes.c_void_p * 9)(
            ctypes.addressof(p_W), ctypes.addressof(p_x), ctypes.addressof(p_y),
            ctypes.addressof(p_M), ctypes.addressof(p_N), ctypes.addressof(p_K),
            ctypes.addressof(p_ng), ctypes.addressof(p_Kx), ctypes.addressof(p_rnd),
        )
        if is_seq:
            gx = (M * N + 63) // 64
            rc = h.hipModuleLaunchKernel(fn, gx, 1, 1, 64, 1, 1, 0, self.stream, args, None)
        else:
            rc = h.hipModuleLaunchKernel(fn, N, M, 1, 32, 1, 1, 0, self.stream, args, None)
        if rc != 0:
            return -4
        assert h.hipMemcpyAsync(ctypes.c_void_p(y_ptr), ctypes.c_void_p(self._d_bf16_y), y_bytes, self.HIP_MEMCPY_DEVICE_TO_HOST, self.stream) == 0
        assert h.hipStreamSynchronize(self.stream) == 0
        return 0

    def hv41c_bf16_seq(self, handle: int, x_ptr: int, M: int, ng: int, K_x: int, bf16_round: int, y_ptr: int, ms_out=None) -> int:
        return self._run_bf16(self.fn_bf16_seq, handle, x_ptr, M, ng, K_x, bf16_round, y_ptr, True)

    def hv41c_bf16_w32(self, handle: int, x_ptr: int, M: int, ng: int, K_x: int, bf16_round: int, y_ptr: int, ms_out=None) -> int:
        return self._run_bf16(self.fn_bf16_w32, handle, x_ptr, M, ng, K_x, bf16_round, y_ptr, False)

    def hv41c_sattn_exptable(self, path: bytes, flag: int = 0) -> int:
        raw = open(path.decode("utf-8") if isinstance(path, bytes) else path, "rb").read()
        if self._d_exptable:
            self.hip.hipFree(ctypes.c_void_p(self._d_exptable))
        p = ctypes.c_void_p()
        assert self.hip.hipMalloc(ctypes.byref(p), len(raw)) == 0
        buf = ctypes.create_string_buffer(raw, len(raw))
        assert self.hip.hipMemcpy(p, buf, len(raw), self.HIP_MEMCPY_HOST_TO_DEVICE) == 0
        self._d_exptable = p.value
        self._exptable_count = len(raw) // 8
        return 0

    def hv41c_sattn(self, Q_ptr: int, M: int, H: int, d: int, K_ptr: int, kv_len: int, idx_ptr: int, topk: int, sink_ptr: int, scale: float, O_ptr: int, ms_out=None) -> int:
        h = self.hip
        q_bytes = M * H * d * 4
        k_bytes = kv_len * d * 4
        idx_bytes = M * topk * 4
        sink_bytes = H * 4
        dQ, dK, dI, dS, dO = [ctypes.c_void_p() for _ in range(5)]
        assert h.hipMalloc(ctypes.byref(dQ), q_bytes) == 0
        assert h.hipMalloc(ctypes.byref(dK), k_bytes) == 0
        assert h.hipMalloc(ctypes.byref(dI), idx_bytes) == 0
        assert h.hipMalloc(ctypes.byref(dS), sink_bytes) == 0
        assert h.hipMalloc(ctypes.byref(dO), q_bytes) == 0
        assert h.hipMemcpy(dQ, ctypes.c_void_p(Q_ptr), q_bytes, self.HIP_MEMCPY_HOST_TO_DEVICE) == 0
        assert h.hipMemcpy(dK, ctypes.c_void_p(K_ptr), k_bytes, self.HIP_MEMCPY_HOST_TO_DEVICE) == 0
        assert h.hipMemcpy(dI, ctypes.c_void_p(idx_ptr), idx_bytes, self.HIP_MEMCPY_HOST_TO_DEVICE) == 0
        assert h.hipMemcpy(dS, ctypes.c_void_p(sink_ptr), sink_bytes, self.HIP_MEMCPY_HOST_TO_DEVICE) == 0

        p_Q = ctypes.c_void_p(dQ.value)
        p_K = ctypes.c_void_p(dK.value)
        p_idx = ctypes.c_void_p(dI.value)
        p_V = ctypes.c_void_p(dK.value)
        p_sink = ctypes.c_void_p(dS.value)
        p_O = ctypes.c_void_p(dO.value)
        p_tab = ctypes.c_void_p(self._d_exptable or 0)
        p_tcnt = ctypes.c_int(self._exptable_count)
        p_M = ctypes.c_int(M)
        p_H = ctypes.c_int(H)
        p_d = ctypes.c_int(d)
        p_kv = ctypes.c_int(kv_len)
        p_topk = ctypes.c_int(topk)
        p_scale = ctypes.c_float(scale)
        args = (ctypes.c_void_p * 14)(
            ctypes.addressof(p_Q), ctypes.addressof(p_K), ctypes.addressof(p_idx),
            ctypes.addressof(p_V), ctypes.addressof(p_sink), ctypes.addressof(p_O),
            ctypes.addressof(p_tab), ctypes.addressof(p_tcnt),
            ctypes.addressof(p_M), ctypes.addressof(p_H), ctypes.addressof(p_d),
            ctypes.addressof(p_kv), ctypes.addressof(p_topk), ctypes.addressof(p_scale),
        )
        rc = h.hipModuleLaunchKernel(self.fn_sattn, H, M, 1, 64, 1, 1, 0, self.stream, args, None)
        assert rc == 0
        assert h.hipMemcpy(ctypes.c_void_p(O_ptr), dO, q_bytes, self.HIP_MEMCPY_DEVICE_TO_HOST) == 0
        for ptr in (dQ, dK, dI, dS, dO):
            h.hipFree(ptr)
        return 0

    def hv41c_resident(self) -> int:
        return self._resident_bytes

    def hv41c_vram_free(self) -> int:
        vfree, vtotal = ctypes.c_size_t(0), ctypes.c_size_t(0)
        if self.hip.hipMemGetInfo(ctypes.byref(vfree), ctypes.byref(vtotal)) != 0:
            return 0
        return vfree.value


def load_hv41cuda(dll_path: str):
    if dll_path == "__direct_only__":
        cpp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hv41cuda_hip.cpp")
        return DirectHipHv41c(cpp_path)
    prom_bin = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(dll_path))),
        "release_0.2",
        "extracted",
        "PROMETHEVS_0.2",
        "bin",
    )
    if os.path.isdir(prom_bin):
        os.add_dll_directory(prom_bin)
    dll_dir = os.path.dirname(os.path.abspath(dll_path))
    os.add_dll_directory(dll_dir)

    try:
        dll = ctypes.WinDLL(dll_path)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            cpp_path = os.path.join(dll_dir, "hv41cuda_hip.cpp")
            print(f"[SAC] WinError 4551 blocked loading {dll_path}; falling back to signed WHQL DirectHipHv41c ({cpp_path})")
            return DirectHipHv41c(cpp_path)
        raise

    dll.hv41c_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    dll.hv41c_init.restype = ctypes.c_int

    dll.hv41c_upload.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    dll.hv41c_upload.restype = ctypes.c_void_p

    dll.hv41c_linear.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.hv41c_linear.restype = ctypes.c_int

    dll.hv41c_resident.argtypes = []
    dll.hv41c_resident.restype = ctypes.c_size_t

    dll.hv41c_vram_free.argtypes = []
    dll.hv41c_vram_free.restype = ctypes.c_size_t

    dll.hv41c_error.argtypes = []
    dll.hv41c_error.restype = ctypes.c_char_p

    if hasattr(dll, "hv41c_upload_bf16"):
        dll.hv41c_upload_bf16.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        dll.hv41c_upload_bf16.restype = ctypes.c_void_p
        for fn_name in ("hv41c_bf16_seq", "hv41c_bf16_w32", "hv41c_bf16_w32t", "hv41c_bf16_w32m"):
            fn = getattr(dll, fn_name)
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ]
            fn.restype = ctypes.c_int
        dll.hv41c_seq_upload_f32.argtypes = [ctypes.c_void_p, ctypes.c_int]
        dll.hv41c_seq_upload_f32.restype = ctypes.c_void_p
        dll.hv41c_sattn_exptable.argtypes = [ctypes.c_char_p, ctypes.c_int]
        dll.hv41c_sattn_exptable.restype = ctypes.c_int
        dll.hv41c_sattn_table_count.argtypes = []
        dll.hv41c_sattn_table_count.restype = ctypes.c_int
        dll.hv41c_sattn.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ]
        dll.hv41c_sattn.restype = ctypes.c_int
    if hasattr(dll, "hv41c_seq_shared"):
        dll.hv41c_seq_shared.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ]
        dll.hv41c_seq_shared.restype = ctypes.c_int
        dll.hv41c_seq_back.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ]
        dll.hv41c_seq_back.restype = ctypes.c_int
        dll.hv41c_seq_front.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ]
        dll.hv41c_seq_front.restype = ctypes.c_int
    return dll


def run_edge_case_tests(dll) -> int:
    """Directed tests for dot_to_scaled_f32 boundary conditions (subnormal -23/-24, ties, 0, inf)."""
    N, K = 64, 64
    nkb = K >> 5  # 2

    codes = np.zeros((N, K), dtype=np.uint8)
    # code 0x38 = exp=7, mant=0 -> mag = 8 << 6 = 512 (exact factor 2^9; divided by 2^18 -> 2^-9)
    # code 0xB8 = negative 512
    # code 0x01 = +1, code 0x81 = -1
    codes[0:32, 0] = 0x01
    codes[32:64, 0] = 0x81
    codes[0:32, 32] = 0x38
    codes[32:64, 32] = 0xB8
    # Additional arbitrary weights in col 1..31
    rng = np.random.default_rng(12345)
    codes[1:32, 1:32] = rng.integers(0, 256, size=(31, 31), dtype=np.uint8)
    codes[33:64, 1:32] = rng.integers(0, 256, size=(31, 31), dtype=np.uint8)

    # e_w edge exponents:
    # row 0 block (cols 0..31): kb=0 uses e_w=-128 (so dot*2^-146), kb=1 uses e_w=127 (overflow when dot large)
    # Wait: keep s tiny or dot controlled to check subnormals first, and another test for overflow!
    scales = np.array([[-128, -110], [-127, 127]], dtype=np.int8)

    # Subnormal exact tests on kb=0 with e_w=-128 (dot * 2^-146):
    # To hit f32_exp = -23 (range [2^-150, 2^-149)), we cannot get 2^-150 from integer*2^-146,
    # Wait: use e_w=-128 -> scale is 2^-146. With e_w=-128 on dot=1 we get 2^-146 (f32_exp=-19).
    # Wait: dot=1, e_w=-128 hits f32_exp = 0 + (-128) + 109 = -19 -> subnormal!
    # Wait: with dot_to_scaled_f32, f32_exp = msb + e_w + 109:
    # For msb=0 (dot=1), e_w=-128 gives f32_exp=-19; for overflow e_w=127 with large dot gives Inf.
    # Let's test multiple exponent combos in two edge passes!
    mismatches = 0
    for case_idx, (e00, e01, e10, e11, a_scale) in enumerate(
        [
            (-128, -120, -127, -121, 1),
            (-128, -20, -126, -15, 7),
            (127, 127, 0, -128, 1000000),
            (-10, -10, -10, -10, 0),
        ]
    ):
        sc_case = np.array([[e00, e01], [e10, e11]], dtype=np.int8)
        h = dll.hv41c_upload(codes.ctypes.data, sc_case.ctypes.data, N, K)
        if not h:
            raise RuntimeError(f"Edge upload failed: {dll.hv41c_error().decode()}")

        for rows in BATCH_ROWS:
            a = rng.integers(-a_scale, a_scale + 1, size=(rows, K), dtype=np.int32)
            if a_scale == 1:
                # Set lane 0 to exact -1, 0, 1, 2, 3 for exact subnormal parity checks
                for r in range(rows):
                    a[r, :] = 0
                    a[r, 0] = [-1, 0, 1, 2, 3, 4][r % 6]
                    a[r, 32] = [-2, 0, 1, 5, -7, 8][r % 6]
            s = rng.uniform(0.5, 1.5, size=(rows, nkb)).astype(np.float32)
            out_gpu = np.empty((rows, N), dtype=np.float32)
            ms = ctypes.c_float(0.0)
            rc = dll.hv41c_linear(
                h, a.ctypes.data, s.ctypes.data, rows, out_gpu.ctypes.data, ctypes.byref(ms)
            )
            if rc != 0:
                raise RuntimeError(f"Edge hv41c_linear failed ({rc}): {dll.hv41c_error().decode()}")

            ref_u32 = cpu_ref_hv41f_exact_i64(codes, sc_case, a, s)
            gpu_u32 = out_gpu.view(np.uint32)
            diff = int(np.sum(gpu_u32 != ref_u32))
            if diff != 0:
                idx = np.argwhere(gpu_u32 != ref_u32)[0]
                r0, c0 = int(idx[0]), int(idx[1])
                print(
                    f"[EDGE FAIL] case={case_idx} rows={rows} diff={diff} "
                    f"at ({r0},{c0}): gpu=0x{gpu_u32[r0,c0]:08x} ref=0x{ref_u32[r0,c0]:08x}"
                )
            mismatches += diff
    if hasattr(dll, "reset_resident"):
        dll.reset_resident()
    print(f"Edge boundary checks (4 cases x 6 batch sizes): mismatches={mismatches}")
    return mismatches


def main() -> int:
    dll_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "hv41cuda.dll")
    )
    dll = load_hv41cuda(dll_path)

    info_buf = ctypes.create_string_buffer(512)
    rc = dll.hv41c_init(info_buf, len(info_buf))
    if rc != 0:
        err = dll.hv41c_error()
        print(f"hv41c_init failed ({rc}): {err.decode() if err else ''}")
        return 1

    print(f"Backend    : {getattr(dll, 'backend_name', dll_path)}")
    print(f"Init info  : {info_buf.value.decode()}")
    print(f"VRAM free  : {dll.hv41c_vram_free() / (1024.0 * 1024.0):.2f} MiB")

    total_mismatches = 0
    total_mismatches += run_edge_case_tests(dll)

    # Cross-check fast BLAS reference against exact int64 reference on tiny_edge first
    rng_chk = np.random.default_rng(42)
    c_chk = rng_chk.integers(0, 256, size=(128, 256), dtype=np.uint8)
    sc_chk = rng_chk.integers(-28, -5, size=(128 // 32, 256 // 32), dtype=np.int8)
    a_chk = rng_chk.integers(-4000, 4001, size=(16, 256), dtype=np.int32)
    s_chk = (rng_chk.choice([-1.0, 1.0], size=(16, 8)) * rng_chk.uniform(0.25, 2.5, size=(16, 8))).astype(np.float32)
    u_i64 = cpu_ref_hv41f_exact_i64(c_chk, sc_chk, a_chk, s_chk)
    u_f64 = cpu_ref_hv41f_fast(c_chk, sc_chk, a_chk, s_chk)
    ref_diff = int(np.sum(u_i64 != u_f64))
    if ref_diff != 0:
        raise AssertionError(f"Internal reference mismatch i64 vs f64: {ref_diff}")

    total_outputs = 0
    print()
    print(
        f"{'Shape':<16} {'N':>5} {'K':>5} {'Rows':>4} | {'Ref ms':>8} {'GPU mean ms':>11} {'GPU min ms':>10} {'Mean GB/s':>9} {'Peak GB/s':>9} | {'Mismatches':>10}"
    )
    print("-" * 104)

    for shape_idx, (name, N, K) in enumerate(SHAPES):
        rng = np.random.default_rng(0x41F000 + shape_idx)
        nkb = K >> 5
        codes = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
        scales = rng.integers(-28, -5, size=(N // 32, nkb), dtype=np.int8)

        handle = dll.hv41c_upload(codes.ctypes.data, scales.ctypes.data, N, K)
        if not handle:
            raise RuntimeError(f"hv41c_upload({name}) failed: {dll.hv41c_error().decode()}")

        for rows in BATCH_ROWS:
            a = rng.integers(-4000, 4001, size=(rows, K), dtype=np.int32)
            s = (
                rng.choice([-1.0, 1.0], size=(rows, nkb))
                * rng.uniform(0.25, 2.5, size=(rows, nkb))
            ).astype(np.float32)

            out_gpu = np.empty((rows, N), dtype=np.float32)
            ms_c = ctypes.c_float(0.0)

            t0 = time.perf_counter()
            ref_u32 = cpu_ref_hv41f_fast(codes, scales, a, s)
            ref_ms = (time.perf_counter() - t0) * 1000.0

            rc = dll.hv41c_linear(
                handle, a.ctypes.data, s.ctypes.data, rows, out_gpu.ctypes.data, ctypes.byref(ms_c)
            )
            if rc != 0:
                raise RuntimeError(
                    f"hv41c_linear({name}, rows={rows}) failed ({rc}): {dll.hv41c_error().decode()}"
                )

            gpu_u32 = out_gpu.view(np.uint32)
            diff = int(np.sum(gpu_u32 != ref_u32))
            total_mismatches += diff
            total_outputs += rows * N

            if diff != 0:
                idx = np.argwhere(gpu_u32 != ref_u32)[0]
                r0, c0 = int(idx[0]), int(idx[1])
                print(
                    f"MISMATCH {name} rows={rows}: count={diff} first=({r0},{c0}) "
                    f"gpu=0x{gpu_u32[r0,c0]:08x} ref=0x{ref_u32[r0,c0]:08x}"
                )

            # Warmup x2 + benchmark x5 steady-state kernel ms
            for _ in range(2):
                dll.hv41c_linear(handle, a.ctypes.data, s.ctypes.data, rows, out_gpu.ctypes.data, ctypes.byref(ms_c))
            times_ms = []
            for _ in range(5):
                rc = dll.hv41c_linear(handle, a.ctypes.data, s.ctypes.data, rows, out_gpu.ctypes.data, ctypes.byref(ms_c))
                if rc != 0:
                    raise RuntimeError(f"bench failed: {dll.hv41c_error().decode()}")
                times_ms.append(float(ms_c.value))

            mean_ms = sum(times_ms) / len(times_ms)
            min_ms = min(times_ms)
            total_bytes = (
                N * K
                + (N // 32) * nkb
                + rows * K * 4
                + rows * nkb * 4
                + rows * N * 4
            )
            mean_gbps = total_bytes / (mean_ms * 1e6) if mean_ms > 0.0 else 0.0
            peak_gbps = total_bytes / (min_ms * 1e6) if min_ms > 0.0 else 0.0

            print(
                f"{name:<16} {N:>5} {K:>5} {rows:>4} | {ref_ms:>8.3f} {mean_ms:>11.4f} {min_ms:>10.4f} {mean_gbps:>9.2f} {peak_gbps:>9.2f} | {diff:>10}"
            )

    print("-" * 104)
    print(f"Resident VRAM weights : {dll.hv41c_resident() / (1024.0 * 1024.0):.2f} MiB ({dll.hv41c_resident()} bytes)")
    print(f"VRAM free remaining   : {dll.hv41c_vram_free() / (1024.0 * 1024.0):.2f} MiB")
    print(f"Total outputs checked : {total_outputs:,}")
    print(f"Total bitwise mismatches: {total_mismatches}")

    if hasattr(dll, "hv41c_upload_bf16"):
        p2_out, p2_bad = verify_phase2_kernels(dll)
        total_outputs += p2_out
        total_mismatches += p2_bad
        print(f"Phase 2 outputs checked : {p2_out:,}, bitwise mismatches: {p2_bad}")

    return 0 if total_mismatches == 0 else 2


def _bf16_u16_to_f32(w_u16: np.ndarray) -> np.ndarray:
    return (w_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _cpu_bf16_seq_ref(w_u16: np.ndarray, x_f32: np.ndarray, M: int, N: int, K: int, ng: int, K_x: int, bf16_round: int) -> np.ndarray:
    w_f64 = _bf16_u16_to_f32(w_u16).astype(np.float64)
    y = np.empty((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            g = n // ng
            x_sl = x_f32[m, g * K:(g + 1) * K].astype(np.float64)
            acc = 0.0
            for k in range(K):
                acc += float(x_sl[k]) * float(w_f64[n, k])
            y[m, n] = np.float32(acc)
    u = y.view(np.uint32)
    if bf16_round:
        u = round_bf16_in_f32_u32(y)
    return u


def _cpu_bf16_w32_ref(w_u16: np.ndarray, x_f32: np.ndarray, M: int, N: int, K: int, ng: int, K_x: int, bf16_round: int) -> np.ndarray:
    w_f32 = _bf16_u16_to_f32(w_u16)
    y = np.empty((M, N), dtype=np.float32)
    nblocks = K // 8
    for m in range(M):
        for n in range(N):
            g = n // ng
            x_sl = x_f32[m, g * K:(g + 1) * K]
            w_sl = w_f32[n]
            buf = [np.float32(0.0)] * 32
            for lane in range(32):
                acc = np.float32(0.0)
                for b in range(lane, nblocks, 32):
                    base = b * 8
                    for t in range(8):
                        acc = np.float32(float(x_sl[base + t]) * float(w_sl[base + t]) + float(acc))
                buf[lane] = acc
            stride = 16
            while stride >= 1:
                for j in range(stride):
                    buf[j] = np.float32(buf[j + stride] + buf[j])
                stride >>= 1
            y[m, n] = buf[0]
    u = y.view(np.uint32)
    if bf16_round:
        u = round_bf16_in_f32_u32(y)
    return u


def _cpu_sattn_ref(Q: np.ndarray, K_mat: np.ndarray, idx: np.ndarray, sink: np.ndarray, scale: float) -> np.ndarray:
    ucrt = ctypes.CDLL("ucrtbase.dll")
    ucrt.expf.argtypes = [ctypes.c_float]
    ucrt.expf.restype = ctypes.c_float

    def c_expf(val: float) -> np.float32:
        return np.float32(ucrt.expf(ctypes.c_float(float(val))))

    M, H, d = Q.shape
    kv_len = K_mat.shape[0]
    topk = idx.shape[1]
    O = np.empty((M, H, d), dtype=np.float32)
    for m in range(M):
        for h in range(H):
            q = Q[m, h]
            o_acc = np.zeros(d, dtype=np.float32)
            m_max = np.float32(-1e30)
            l_sum = np.float32(0.0)
            for c0 in range(0, topk, 64):
                c_len = min(64, topk - c0)
                scores = np.empty(c_len, dtype=np.float32)
                for t in range(c_len):
                    tok = int(idx[m, c0 + t])
                    if tok < 0 or tok >= kv_len:
                        scores[t] = np.float32(-np.inf)
                    else:
                        dot = np.float32(0.0)
                        k_row = K_mat[tok]
                        for k in range(d):
                            dot = np.float32(dot + np.float32(q[k] * k_row[k]))
                        scores[t] = np.float32(dot * np.float32(scale))
                m_old = m_max
                m_new = m_old
                for t in range(c_len):
                    if scores[t] > m_new:
                        m_new = scores[t]
                m_max = m_new
                alpha = c_expf(np.float32(m_old - m_new))
                p_arr = np.empty(c_len, dtype=np.float32)
                p_bf16 = np.empty(c_len, dtype=np.float32)
                psum = np.float32(0.0)
                for t in range(c_len):
                    p = c_expf(np.float32(scores[t] - m_new))
                    p_arr[t] = p
                    p_bf16[t] = round_bf16_in_f32_u32(np.array([p], dtype=np.float32)).view(np.float32)[0]
                    psum = np.float32(psum + p)
                l_sum = np.float32(np.float32(alpha * l_sum) + psum)
                for j in range(d):
                    acc = np.float32(alpha * o_acc[j])
                    for t in range(c_len):
                        tok = int(idx[m, c0 + t])
                        if 0 <= tok < kv_len:
                            acc = np.float32(acc + np.float32(p_bf16[t] * K_mat[tok, j]))
                    o_acc[j] = acc
            l_final = np.float32(l_sum + c_expf(np.float32(sink[h] - m_max)))
            for j in range(d):
                O[m, h, j] = np.float32(o_acc[j] / l_final)
    return round_bf16_in_f32_u32(O)


def verify_phase2_kernels(dll) -> tuple[int, int]:
    rng = np.random.default_rng(20261118)
    total_out = 0
    total_bad = 0
    print("\n=== Phase 2 Optional GPU Kernels Bitwise Check ===")

    # 1. BF16 seq + w32 checks
    for (M, N, K, ng, rnd) in [(1, 64, 256, 1, 1), (3, 128, 256, 4, 0), (2, 64, 512, 64, 1)]:
        groups = (N - 1) // ng + 1
        K_x = groups * K
        w_f32 = rng.standard_normal((N, K), dtype=np.float32) * 0.25
        w_u16 = (round_bf16_in_f32_u32(w_f32) >> np.uint32(16)).astype(np.uint16)
        x_f32 = rng.standard_normal((M, K_x), dtype=np.float32) * 0.25
        h_bf16 = dll.hv41c_upload_bf16(w_u16.ctypes.data, N, K)
        assert h_bf16 is not None

        y_seq = np.empty((M, N), dtype=np.float32)
        assert dll.hv41c_bf16_seq(h_bf16, x_f32.ctypes.data, M, ng, K_x, rnd, y_seq.ctypes.data, None) == 0
        ref_seq = _cpu_bf16_seq_ref(w_u16, x_f32, M, N, K, ng, K_x, rnd)
        bad_seq = int(np.count_nonzero(y_seq.view(np.uint32) != ref_seq))
        print(f"  bf16_seq  M={M} N={N} K={K} ng={ng} rnd={rnd}: outputs={M*N}, mismatches={bad_seq}")
        total_out += M * N
        total_bad += bad_seq

        y_w32 = np.empty((M, N), dtype=np.float32)
        assert dll.hv41c_bf16_w32(h_bf16, x_f32.ctypes.data, M, ng, K_x, rnd, y_w32.ctypes.data, None) == 0
        ref_w32 = _cpu_bf16_w32_ref(w_u16, x_f32, M, N, K, ng, K_x, rnd)
        bad_w32 = int(np.count_nonzero(y_w32.view(np.uint32) != ref_w32))
        print(f"  bf16_w32  M={M} N={N} K={K} ng={ng} rnd={rnd}: outputs={M*N}, mismatches={bad_w32}")
        total_out += M * N
        total_bad += bad_w32

    # 2. Sparse Attention (hv41c_sattn) with exptable.bin
    exptab_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "release_0.2", "extracted", "PROMETHEVS_0.2", "data", "exptable.bin",
    )
    if os.path.exists(exptab_path):
        dll.hv41c_sattn_exptable(exptab_path.encode("utf-8"), 0)
    for (M, H, d, kv_len, topk) in [(1, 8, 64, 128, 96), (2, 4, 512, 96, 80), (31, 8, 512, 64, 32)]:
        Q = rng.standard_normal((M, H, d), dtype=np.float32) * 0.1
        K_mat = rng.standard_normal((kv_len, d), dtype=np.float32) * 0.1
        idx = rng.integers(-1, kv_len, size=(M, topk), dtype=np.int32)
        idx[:, 0] = -1
        sink = rng.standard_normal(H, dtype=np.float32) * 0.5
        O_gpu = np.empty((M, H, d), dtype=np.float32)
        assert dll.hv41c_sattn(
            Q.ctypes.data, M, H, d, K_mat.ctypes.data, kv_len,
            idx.ctypes.data, topk, sink.ctypes.data, 0.125, O_gpu.ctypes.data, None,
        ) == 0
        ref_o = _cpu_sattn_ref(Q, K_mat, idx, sink, 0.125)
        bad_sa = int(np.count_nonzero(O_gpu.view(np.uint32) != ref_o))
        print(f"  sattn     M={M} H={H} d={d} kv={kv_len} topk={topk}: outputs={M*H*d}, mismatches={bad_sa}")
        total_out += M * H * d
        total_bad += bad_sa

    # 3. Fused seq_shared, seq_back, seq_front
    if hasattr(dll, "hv41c_seq_shared"):
        def _cpu_act_quant32(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            K = len(x)
            av = np.empty(K, dtype=np.int32)
            sv = np.empty(K // 32, dtype=np.float32)
            for g in range(K // 32):
                blk = x[g * 32 : (g + 1) * 32]
                amax = float(max(1e-4, float(np.max(np.abs(blk)))))
                s_raw = np.float32(amax * float(np.uint32(0x3B124925).view(np.float32)))
                u = int(s_raw.view(np.uint32))
                e = ((u >> 23) & 0xFF) + (1 if (u & 0x7FFFFF) != 0 else 0)
                sc = np.uint32(e << 23).view(np.float32)
                sv[g] = sc
                for i in range(32):
                    v = np.float32(blk[i] / sc)
                    if not np.isnan(v):
                        v = np.float32(max(-448.0, min(448.0, float(v))))
                        if v != 0.0:
                            ab = abs(float(v))
                            eu = ((int(np.float32(ab).view(np.uint32)) >> 23) & 0xFF) - 127
                            eq = max(-6, eu)
                            step = float(np.uint32((eq + 124) << 23).view(np.float32))
                            mag = min(448.0, float(np.rint(ab / step)) * step)
                            v = np.float32(-mag if v < 0 else mag)
                    av[g * 32 + i] = int(np.float32(v) * np.float32(512.0))
            return av, sv

        def _up_fp8(N: int, K: int):
            c = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
            c &= np.uint8(0xBF)
            s = rng.integers(-6, 4, size=(N // 32, K // 32), dtype=np.int8)
            h = dll.hv41c_upload(c.ctypes.data, s.ctypes.data, N, K)
            assert h
            return h, c, s

        # seq_shared
        K_in, N_mid, N_out = 256, 128, 256
        h_g, cg, sg = _up_fp8(N_mid, K_in)
        h_u, cu, su = _up_fp8(N_mid, K_in)
        h_d, cd, sd = _up_fp8(N_out, N_mid)
        av_in = rng.integers(-200, 201, size=(1, K_in), dtype=np.int32)
        sv_in = rng.uniform(0.5, 1.5, size=(1, K_in // 32)).astype(np.float32)
        y_shx = np.empty(N_out, dtype=np.float32)
        assert dll.hv41c_seq_shared(av_in.ctypes.data, sv_in.ctypes.data, h_g, h_u, h_d, y_shx.ctypes.data, None) == 0
        rg = cpu_ref_hv41f_exact_i64(cg, sg, av_in, sv_in).view(np.float32)[0]
        ru = cpu_ref_hv41f_exact_i64(cu, su, av_in, sv_in).view(np.float32)[0]
        mid_f32 = np.empty(N_mid, dtype=np.float32)
        for i in range(N_mid):
            gv = min(10.0, float(rg[i]))
            uv = max(-10.0, min(10.0, float(ru[i])))
            silu = np.float32(gv / (1.0 + np.exp(-np.float32(gv))))
            mid_f32[i] = round_bf16_in_f32_u32(np.array([np.float32(silu * np.float32(uv))], dtype=np.float32)).view(np.float32)[0]
        av_m, sv_m = _cpu_act_quant32(mid_f32)
        ref_shx = cpu_ref_hv41f_exact_i64(cd, sd, av_m.reshape(1, -1), sv_m.reshape(1, -1))[0]
        bad_shx = int(np.count_nonzero(y_shx.view(np.uint32) != ref_shx))
        print(f"  seq_shared K={K_in} mid={N_mid} out={N_out}: outputs={N_out}, mismatches={bad_shx}")
        total_out += N_out
        total_bad += bad_shx

        # seq_back
        w_f32 = rng.standard_normal((128, 256), dtype=np.float32) * 0.25
        w_u16 = (round_bf16_in_f32_u32(w_f32) >> np.uint32(16)).astype(np.uint16)
        h_bf = dll.hv41c_upload_bf16(w_u16.ctypes.data, 128, 256)
        x_b = rng.standard_normal((1, 512), dtype=np.float32) * 0.25
        h_bo, cbo, sbo = _up_fp8(64, 128)
        y_bk = np.empty(64, dtype=np.float32)
        assert dll.hv41c_seq_back(h_bf, x_b.ctypes.data, 64, 512, 1, 1, h_bo, y_bk.ctypes.data, None) == 0
        ref_mid_b = _cpu_bf16_w32_ref(w_u16, x_b, 1, 128, 256, 64, 512, 1).view(np.float32)[0]
        av_b, sv_b = _cpu_act_quant32(ref_mid_b)
        ref_bk = cpu_ref_hv41f_exact_i64(cbo, sbo, av_b.reshape(1, -1), sv_b.reshape(1, -1))[0]
        bad_bk = int(np.count_nonzero(y_bk.view(np.uint32) != ref_bk))
        print(f"  seq_back  N_mid=128 out=64: outputs=64, mismatches={bad_bk}")
        total_out += 64
        total_bad += bad_bk

        # seq_front
        h_kv, ckv, skv = _up_fp8(64, 256)
        h_qa, cqa, sqa = _up_fp8(128, 256)
        h_qb, cqb, sqb = _up_fp8(128, 128)
        qn_host = rng.uniform(0.5, 1.5, size=128).astype(np.float32)
        qn_dev = dll.hv41c_seq_upload_f32(qn_host.ctypes.data, 128)
        assert qn_dev
        y_kv = np.empty(64, dtype=np.float32)
        y_qa = np.empty(128, dtype=np.float32)
        y_qb = np.empty(128, dtype=np.float32)
        w3_arr = (ctypes.c_void_p * 2)(h_kv, h_qa)
        wq_arr = (ctypes.c_void_p * 2)(h_qb, None)
        op_arr = (ctypes.c_void_p * 2)(y_kv.ctypes.data, None)
        yqp_arr = (ctypes.c_void_p * 2)(y_qb.ctypes.data, None)
        assert dll.hv41c_seq_front(
            av_in.ctypes.data, sv_in.ctypes.data, w3_arr, 2, 1,
            qn_dev, np.float32(1e-6), wq_arr, 1,
            op_arr, y_qa.ctypes.data, yqp_arr, None,
        ) == 0
        ref_kv = cpu_ref_hv41f_exact_i64(ckv, skv, av_in, sv_in)[0]
        raw_qa = cpu_ref_hv41f_exact_i64(cqa, sqa, av_in, sv_in).view(np.float32)[0]
        rms_inv = np.float32(1.0) / np.float32(np.sqrt(float(np.sum(raw_qa * raw_qa)) / 128.0 + 1e-6))
        ref_qa_u32 = round_bf16_in_f32_u32((rms_inv * raw_qa * qn_host).astype(np.float32))
        av_q, sv_q = _cpu_act_quant32(ref_qa_u32.view(np.float32))
        ref_qb = cpu_ref_hv41f_exact_i64(cqb, sqb, av_q.reshape(1, -1), sv_q.reshape(1, -1))[0]
        bad_fr = (
            int(np.count_nonzero(y_kv.view(np.uint32) != ref_kv))
            + int(np.count_nonzero(y_qa.view(np.uint32) != ref_qa_u32))
            + int(np.count_nonzero(y_qb.view(np.uint32) != ref_qb))
        )
        print(f"  seq_front kv=64 qa=128 qb=128: outputs={64+128+128}, mismatches={bad_fr}")
        total_out += 64 + 128 + 128
        total_bad += bad_fr

    # corr: hv41_plane_restore_kernel + hv41_fp4_linear_kernel (tested via DirectHipHv41c or dll)
    if isinstance(dll, DirectHipHv41c):
        N_c, K_c, M_c, zs_c = 64, 128, 3, 1
        G_n, K_blk = N_c >> 3, K_c >> 5
        G_tot = G_n * K_blk
        p0 = rng.integers(0, 256, size=(G_tot, 64), dtype=np.uint8)
        p1 = rng.integers(0, 256, size=(G_tot, 64), dtype=np.uint8)
        tbl = rng.integers(0, 256, size=(G_tot, 19), dtype=np.uint8)
        field = np.concatenate([p0.reshape(-1), tbl.reshape(-1)])
        plane1 = p1.reshape(-1)
        lut32 = rng.integers(0, 16, size=32768, dtype=np.uint8)
        patch_idx = np.arange(G_n + 1, dtype=np.uint32) * np.uint32(1024)
        patch_bytes = rng.integers(0, 256, size=(G_n + 2) * 256, dtype=np.uint8)

        # CPU reference for plane_restore
        ref_codes = np.zeros((N_c, K_c >> 1), dtype=np.uint8)
        ref_scales = np.zeros((N_c, K_blk), dtype=np.uint8)
        def _read_bits(buf, bp, nb):
            bidx = bp >> 3
            sh = bp & 7
            w = int.from_bytes(buf[bidx:bidx + 4].tobytes(), "little")
            return (w >> sh) & ((1 << nb) - 1)
        for g in range(G_n):
            bp = int(patch_idx[g])
            for kb in range(K_blk):
                gi = g * K_blk + kb
                t = tbl[gi]
                q0 = int.from_bytes(t[0:8].tobytes(), "little")
                q6 = int.from_bytes(t[6:14].tobytes(), "little")
                idxs = [
                    q0 & 0x7FF, (q0 >> 11) & 0x7FF, (q0 >> 22) & 0x7FF, (q0 >> 33) & 0x7FF,
                    (q0 >> 44) & 0x7FF, (q6 >> 7) & 0x7FF, (q6 >> 18) & 0x7FF, (q6 >> 29) & 0x7FF,
                ]
                for r in range(8):
                    ref_scales[g * 8 + r, kb] = t[11 + r]
                raw_rows = []
                for h_idx in range(2):
                    a0 = p0[gi, h_idx * 32 : (h_idx + 1) * 32]
                    a1 = p1[gi, h_idx * 32 : (h_idx + 1) * 32]
                    ym5 = (((a1 & 0x33) << 2) | (a0 & 0x33)).astype(np.uint8)
                    ym2 = (((a0 >> 2) & 0x33) | (a1 & 0xCC)).astype(np.uint8)
                    for q in range(4):
                        raw_rows.append(np.concatenate([ym5[q * 8 : (q + 1) * 8], ym2[q * 8 : (q + 1) * 8]]))
                for r in range(8):
                    lut = lut32[idxs[r] * 16 : idxs[r] * 16 + 16]
                    row = bytearray(16)
                    mask4 = 0
                    mask1 = 0
                    for i in range(16):
                        b = int(raw_rows[r][i])
                        lo, hi = b & 0xF, (b >> 4) & 0xF
                        olo, ohi = int(lut[lo]), int(lut[hi])
                        row[i] = (ohi << 4) | olo
                        if lo > 11: mask4 |= (1 << i)
                        if hi > 11: mask4 |= (1 << (i + 16))
                        if olo == 0: mask1 |= (1 << i)
                        if ohi == 0: mask1 |= (1 << (i + 16))
                    while mask4:
                        lsb = mask4 & -mask4
                        pos = lsb.bit_length() - 1
                        mask4 ^= lsb
                        nib = _read_bits(patch_bytes, bp, 4)
                        bp += 4
                        bi = pos & 15
                        if pos < 16: row[bi] = (row[bi] & 0xF0) | nib
                        else:        row[bi] = (row[bi] & 0x0F) | (nib << 4)
                    if zs_c:
                        while mask1:
                            lsb = mask1 & -mask1
                            pos = lsb.bit_length() - 1
                            mask1 ^= lsb
                            bit = _read_bits(patch_bytes, bp, 1)
                            bp += 1
                            if bit: row[pos & 15] |= (0x08 if pos < 16 else 0x80)
                    ref_codes[g * 8 + r, kb * 16 : (kb + 1) * 16] = np.frombuffer(row, dtype=np.uint8)

        # CPU reference for fp4_linear (hv41p::dot8)
        av_c = rng.integers(-50000, 50001, size=(M_c, K_c), dtype=np.int32)
        sv_c = rng.uniform(0.5, 1.5, size=(M_c, K_blk)).astype(np.float32)
        bad_c = np.array([0, 1, 0], dtype=np.uint8)
        KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int64)
        ref_y = np.empty((M_c, N_c), dtype=np.float32)
        for m in range(M_c):
            if bad_c[m]:
                ref_y[m, :] = np.uint32(0x7FC00000).view(np.float32)
                continue
            for r in range(N_c):
                acc = np.float32(0.0)
                for kb in range(K_blk):
                    cg = ref_codes[r, kb * 16 : (kb + 1) * 16]
                    ag = av_c[m, kb * 32 : (kb + 1) * 32].astype(np.int64)
                    dot = int(np.sum(KV[cg & 0xF] * ag[:16] + KV[(cg >> 4) & 0xF] * ag[16:]))
                    sc_d = np.uint64((int(ref_scales[r, kb]) + 0x376) << 52).view(np.float64)
                    acc = np.float32(acc + np.float32(np.float64(dot) * sc_d) * sv_c[m, kb])
                ref_y[m, r] = round_bf16_in_f32_u32(np.array([acc], dtype=np.float32)).view(np.float32)[0]

        h_hip = dll.hip
        def _up_dev(arr):
            p = ctypes.c_void_p()
            assert h_hip.hipMalloc(ctypes.byref(p), arr.nbytes) == 0
            assert h_hip.hipMemcpy(p, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, dll.HIP_MEMCPY_HOST_TO_DEVICE) == 0
            return p
        d_f = _up_dev(field); d_p1 = _up_dev(plane1); d_pi = _up_dev(patch_idx)
        d_pb = _up_dev(patch_bytes); d_lut = _up_dev(lut32)
        d_oc = ctypes.c_void_p(); assert h_hip.hipMalloc(ctypes.byref(d_oc), ref_codes.nbytes) == 0
        d_os = ctypes.c_void_p(); assert h_hip.hipMalloc(ctypes.byref(d_os), ref_scales.nbytes) == 0
        vN, vK, vzs = ctypes.c_int(N_c), ctypes.c_int(K_c), ctypes.c_int(zs_c)
        args_pr = (ctypes.c_void_p * 10)(
            ctypes.addressof(d_f), ctypes.addressof(d_p1), ctypes.addressof(d_pi),
            ctypes.addressof(d_pb), ctypes.addressof(d_lut),
            ctypes.addressof(vN), ctypes.addressof(vK), ctypes.addressof(vzs),
            ctypes.addressof(d_oc), ctypes.addressof(d_os),
        )
        assert h_hip.hipModuleLaunchKernel(dll.fn_plane_restore, G_n, 1, 1, 256, 1, 1, 0, dll.stream, args_pr, None) == 0
        gpu_codes = np.empty_like(ref_codes)
        gpu_scales = np.empty_like(ref_scales)
        assert h_hip.hipMemcpy(gpu_codes.ctypes.data_as(ctypes.c_void_p), d_oc, ref_codes.nbytes, dll.HIP_MEMCPY_DEVICE_TO_HOST) == 0
        assert h_hip.hipMemcpy(gpu_scales.ctypes.data_as(ctypes.c_void_p), d_os, ref_scales.nbytes, dll.HIP_MEMCPY_DEVICE_TO_HOST) == 0

        d_av = _up_dev(av_c); d_sv = _up_dev(sv_c); d_bad = _up_dev(bad_c)
        d_y = ctypes.c_void_p(); assert h_hip.hipMalloc(ctypes.byref(d_y), ref_y.nbytes) == 0
        vM = ctypes.c_int(M_c)
        args_f4 = (ctypes.c_void_p * 9)(
            ctypes.addressof(d_oc), ctypes.addressof(d_os),
            ctypes.addressof(d_av), ctypes.addressof(d_sv), ctypes.addressof(d_bad),
            ctypes.addressof(d_y), ctypes.addressof(vM), ctypes.addressof(vN), ctypes.addressof(vK),
        )
        assert h_hip.hipModuleLaunchKernel(dll.fn_fp4_linear, (N_c + 63) // 64, M_c, 1, 64, 1, 1, 0, dll.stream, args_f4, None) == 0
        gpu_y = np.empty_like(ref_y)
        assert h_hip.hipMemcpy(gpu_y.ctypes.data_as(ctypes.c_void_p), d_y, ref_y.nbytes, dll.HIP_MEMCPY_DEVICE_TO_HOST) == 0
        for dp in (d_f, d_p1, d_pi, d_pb, d_lut, d_oc, d_os, d_av, d_sv, d_bad, d_y):
            h_hip.hipFree(dp)

        bad_c_cnt = int(np.count_nonzero(gpu_codes != ref_codes)) + int(np.count_nonzero(gpu_scales != ref_scales))
        bad_y_cnt = int(np.count_nonzero(gpu_y.view(np.uint32) != ref_y.view(np.uint32)))
        n_corr_out = ref_codes.size + ref_scales.size + ref_y.size
        print(f"  corr_restore+fp4 N={N_c} K={K_c} M={M_c}: outputs={n_corr_out}, mismatches={bad_c_cnt + bad_y_cnt}")
        total_out += n_corr_out
        total_bad += bad_c_cnt + bad_y_cnt
    else:
        # Test actual C ABI hv41c_corr_* exports on full 2304x5120 (w1, w3) and 5120x2304 (w2)
        class GcMat(ctypes.Structure):
            _fields_ = [
                ("N", ctypes.c_int), ("K", ctypes.c_int),
                ("b2_off", ctypes.c_uint64), ("pl_off", ctypes.c_uint64),
                ("field_off", ctypes.c_uint64), ("table_off", ctypes.c_uint64),
                ("patch_off", ctypes.c_uint64), ("zs", ctypes.c_uint32),
            ]
        dll.hv41c_corr_open.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
        dll.hv41c_corr_open.restype = ctypes.c_int
        dll.hv41c_corr_host.argtypes = [ctypes.c_int]
        dll.hv41c_corr_host.restype = ctypes.c_void_p
        dll.hv41c_corr_hs_wait.argtypes = [ctypes.c_int]
        dll.hv41c_corr_hs_wait.restype = None
        dll.hv41c_corr_load_gc.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
        dll.hv41c_corr_load_gc.restype = ctypes.c_int
        dll.hv41c_corr_load_fp4.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64)]
        dll.hv41c_corr_load_fp4.restype = ctypes.c_int
        dll.hv41c_corr_decode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(GcMat), ctypes.c_int]
        dll.hv41c_corr_decode.restype = ctypes.c_int
        dll.hv41c_corr_decode_fp4.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        dll.hv41c_corr_decode_fp4.restype = ctypes.c_int
        dll.hv41c_corr_read_decoded.argtypes = [ctypes.c_int, ctypes.c_void_p]
        dll.hv41c_corr_read_decoded.restype = ctypes.c_int
        dll.hv41c_corr_w13.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        dll.hv41c_corr_w13.restype = ctypes.c_int
        dll.hv41c_corr_w2.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        dll.hv41c_corr_w2.restype = ctypes.c_int
        dll.hv41c_corr_wait13.argtypes = [ctypes.c_int]
        dll.hv41c_corr_wait13.restype = ctypes.c_int
        dll.hv41c_corr_wait2.argtypes = [ctypes.c_int]
        dll.hv41c_corr_wait2.restype = ctypes.c_int
        dll.hv41c_corr_yg.argtypes = [ctypes.c_int]
        dll.hv41c_corr_yg.restype = ctypes.POINTER(ctypes.c_float)
        dll.hv41c_corr_yu.argtypes = [ctypes.c_int]
        dll.hv41c_corr_yu.restype = ctypes.POINTER(ctypes.c_float)
        dll.hv41c_corr_y2.argtypes = [ctypes.c_int]
        dll.hv41c_corr_y2.restype = ctypes.POINTER(ctypes.c_float)

        lut32 = rng.integers(0, 16, size=32768, dtype=np.uint8)
        open_info = (ctypes.c_int64 * 4)()
        hs_b = 12 * 1024 * 1024
        dev_b = 24 * 1024 * 1024
        assert dll.hv41c_corr_open(6, hs_b, 2, dev_b, lut32.ctypes.data, 256, open_info) == 0
        assert open_info[2] == 6 * hs_b + 2 * dev_b
        assert open_info[3] == 2

        # Real NVIDIA matrix order: [w1, w2, w3]
        shapes = [(2304, 5120), (5120, 2304), (2304, 5120)]
        pl_bufs, b2_bufs, p0s, p1s, tbls, pidxs, pbytess = [], [], [], [], [], [], []
        for N_c, K_c in shapes:
            G_n, K_blk = N_c >> 3, K_c >> 5
            G_tot = G_n * K_blk
            p0 = rng.integers(0, 256, size=(G_tot, 64), dtype=np.uint8)
            p1 = rng.integers(0, 256, size=(G_tot, 64), dtype=np.uint8)
            tbl = rng.integers(0, 256, size=(G_tot, 19), dtype=np.uint8)
            pidx = np.arange(G_n + 1, dtype=np.uint32) * np.uint32(1024)
            pbytes = rng.integers(0, 256, size=(G_n + 2) * 256, dtype=np.uint8)
            hdr = np.zeros(256, dtype=np.uint8)
            pl = np.concatenate([hdr, p1.reshape(-1), pidx.view(np.uint8), pbytes])
            b2 = np.concatenate([p0.reshape(-1), tbl.reshape(-1)])
            pl_bufs.append(pl); b2_bufs.append(b2)
            p0s.append(p0); p1s.append(p1); tbls.append(tbl); pidxs.append(pidx); pbytess.append(pbytes)

        b2_cat = np.concatenate(b2_bufs)
        # Exercise hs >= dev_cnt and dev_slot != hs (dev_slot=1, hs=4, dec_slot=1, ev/out_slot=5)
        h_ptr = dll.hv41c_corr_host(4)
        assert h_ptr
        ctypes.memmove(h_ptr, b2_cat.ctypes.data, b2_cat.nbytes)
        ptrs_arr = (ctypes.c_void_p * 3)(pl_bufs[0].ctypes.data, pl_bufs[1].ctypes.data, pl_bufs[2].ctypes.data)
        sizes_arr = (ctypes.c_uint64 * 3)(pl_bufs[0].nbytes, pl_bufs[1].nbytes, pl_bufs[2].nbytes)
        assert dll.hv41c_corr_load_gc(1, 4, ptrs_arr, sizes_arr, b2_cat.nbytes) == 0

        pl_tot = sum(x.nbytes for x in pl_bufs)
        mats = (GcMat * 3)()
        b2_cur = pl_tot
        pl_cur = 0
        for i, (N_c, K_c) in enumerate(shapes):
            mats[i] = GcMat(N_c, K_c, b2_cur, pl_cur, 256, 256 + p1s[i].nbytes, 256 + p1s[i].nbytes + pidxs[i].nbytes, 1)
            b2_cur += b2_bufs[i].nbytes
            pl_cur += pl_bufs[i].nbytes

        assert dll.hv41c_corr_decode(1, 1, mats, 5) == 0
        dec_host = np.empty(0x11EE000, dtype=np.uint8)
        assert dll.hv41c_corr_read_decoded(1, dec_host.ctypes.data) == 0

        M_c = 2
        av_13 = rng.integers(-50000, 50001, size=(M_c, 5120), dtype=np.int32)
        sv_13 = rng.uniform(0.5, 1.5, size=(M_c, 160)).astype(np.float32)
        bad_13 = np.array([0, 1], dtype=np.uint8)
        assert dll.hv41c_corr_w13(1, 5, av_13.ctypes.data, sv_13.ctypes.data, bad_13.ctypes.data, M_c) == 0
        assert dll.hv41c_corr_wait13(5) == 0
        yg_gpu = np.ctypeslib.as_array(dll.hv41c_corr_yg(5), shape=(M_c, 2304)).copy()
        yu_gpu = np.ctypeslib.as_array(dll.hv41c_corr_yu(5), shape=(M_c, 2304)).copy()

        av_2 = rng.integers(-50000, 50001, size=(M_c, 2304), dtype=np.int32)
        sv_2 = rng.uniform(0.5, 1.5, size=(M_c, 72)).astype(np.float32)
        bad_2 = np.array([0, 0], dtype=np.uint8)
        assert dll.hv41c_corr_w2(1, 5, av_2.ctypes.data, sv_2.ctypes.data, bad_2.ctypes.data, M_c) == 0
        assert dll.hv41c_corr_wait2(5) == 0
        y2_gpu = np.ctypeslib.as_array(dll.hv41c_corr_y2(5), shape=(M_c, 5120)).copy()

        # Verify decoded codes & scales (slot 0=w1, slot 1=w2, slot 2=w3) + w13/w2 against CPU ref
        KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int64)
        def _cpu_dot8(codes, scales, av, sv, bad, N_c, K_c):
            K_blk = K_c >> 5
            out = np.empty((av.shape[0], N_c), dtype=np.float32)
            for m in range(av.shape[0]):
                if bad[m]:
                    out[m, :] = np.uint32(0x7FC00000).view(np.float32)
                    continue
                acc = np.zeros(N_c, dtype=np.float32)
                for kb in range(K_blk):
                    cg = codes[:, kb * 16 : (kb + 1) * 16]
                    ag = av[m, kb * 32 : (kb + 1) * 32].astype(np.int64)
                    dot = (KV[cg & 0xF] * ag[:16][None, :] + KV[(cg >> 4) & 0xF] * ag[16:][None, :]).sum(axis=1)
                    sc_d = np.ldexp(1.0, scales[:, kb].astype(np.int32) - 137)
                    acc = acc + (dot.astype(np.float64) * sc_d).astype(np.float32) * sv[m, kb]
                out[m, :] = round_bf16_in_f32_u32(acc).view(np.float32)
            return out

        c_w1 = np.ascontiguousarray(dec_host[0 : 0x5A0000].reshape(2304, 2560))
        s_w1 = np.ascontiguousarray(dec_host[0x5A0000 : 0x5FA000].reshape(2304, 160))
        c_w2 = np.ascontiguousarray(dec_host[0x5FA000 : 0x5FA000 + 0x5A0000].reshape(5120, 1152))
        s_w2 = np.ascontiguousarray(dec_host[0x5FA000 + 0x5A0000 : 0xBF4000].reshape(5120, 72))
        c_w3 = np.ascontiguousarray(dec_host[0xBF4000 : 0xBF4000 + 0x5A0000].reshape(2304, 2560))
        s_w3 = np.ascontiguousarray(dec_host[0xBF4000 + 0x5A0000 : 0x11EE000].reshape(2304, 160))

        ref_yg = _cpu_dot8(c_w1, s_w1, av_13, sv_13, bad_13, 2304, 5120)
        ref_yu = _cpu_dot8(c_w3, s_w3, av_13, sv_13, bad_13, 2304, 5120)
        ref_y2 = _cpu_dot8(c_w2, s_w2, av_2, sv_2, bad_2, 5120, 2304)
        bad_corr = (
            int(np.count_nonzero(yg_gpu.view(np.uint32) != ref_yg.view(np.uint32)))
            + int(np.count_nonzero(yu_gpu.view(np.uint32) != ref_yu.view(np.uint32)))
            + int(np.count_nonzero(y2_gpu.view(np.uint32) != ref_y2.view(np.uint32)))
        )
        n_corr = yg_gpu.size + yu_gpu.size + y2_gpu.size
        print(f"  corr_abi w13+w2 M={M_c} (2304x5120x2 + 5120x2304): outputs={n_corr}, mismatches={bad_corr}")
        total_out += n_corr
        total_bad += bad_corr

        # Test FP4 upload > hs_bytes (18.8 MB total > 12 MiB hs_bytes, < 24 MiB dev_bytes)
        fp4_arrs = [c_w1, s_w1, c_w2, s_w2, c_w3, s_w3]
        fp4_ptrs = (ctypes.c_void_p * 6)(*(a.ctypes.data for a in fp4_arrs))
        fp4_sizes = (ctypes.c_uint64 * 6)(*(a.nbytes for a in fp4_arrs))
        assert dll.hv41c_corr_load_fp4(0, fp4_ptrs, fp4_sizes) == 0
        assert dll.hv41c_corr_decode_fp4(0, 0, 7) == 0
        assert dll.hv41c_corr_w13(0, 7, av_13.ctypes.data, sv_13.ctypes.data, bad_13.ctypes.data, M_c) == 0
        assert dll.hv41c_corr_wait13(7) == 0
        yg_fp4 = np.ctypeslib.as_array(dll.hv41c_corr_yg(7), shape=(M_c, 2304)).copy()
        yu_fp4 = np.ctypeslib.as_array(dll.hv41c_corr_yu(7), shape=(M_c, 2304)).copy()
        assert dll.hv41c_corr_w2(0, 7, av_2.ctypes.data, sv_2.ctypes.data, bad_2.ctypes.data, M_c) == 0
        assert dll.hv41c_corr_wait2(7) == 0
        y2_fp4 = np.ctypeslib.as_array(dll.hv41c_corr_y2(7), shape=(M_c, 5120)).copy()
        bad_fp4 = (
            int(np.count_nonzero(yg_fp4.view(np.uint32) != ref_yg.view(np.uint32)))
            + int(np.count_nonzero(yu_fp4.view(np.uint32) != ref_yu.view(np.uint32)))
            + int(np.count_nonzero(y2_fp4.view(np.uint32) != ref_y2.view(np.uint32)))
        )
        n_fp4 = yg_fp4.size + yu_fp4.size + y2_fp4.size
        print(f"  corr_abi fp4 w13+w2 M={M_c} ({sum(a.nbytes for a in fp4_arrs)} B): outputs={n_fp4}, mismatches={bad_fp4}")
        total_out += n_fp4
        total_bad += bad_fp4

        # Async pipelined overlap stress test across 2 concurrent experts (slots 0/8 and 1/9)
        dll.hv41c_corr_take_times.argtypes = [ctypes.POINTER(ctypes.c_double)]
        dll.hv41c_corr_take_times.restype = None

        pl_bufs_b, b2_bufs_b, p1s_b, pidxs_b = [], [], [], []
        for N_c, K_c in shapes:
            G_n, K_blk = N_c >> 3, K_c >> 5
            G_tot = G_n * K_blk
            p0_b = rng.integers(0, 256, size=(G_tot, 64), dtype=np.uint8)
            p1_b = rng.integers(0, 256, size=(G_tot, 64), dtype=np.uint8)
            tbl_b = rng.integers(0, 256, size=(G_tot, 19), dtype=np.uint8)
            pidx_b = np.arange(G_n + 1, dtype=np.uint32) * np.uint32(1024)
            pbytes_b = rng.integers(0, 256, size=(G_n + 2) * 256, dtype=np.uint8)
            hdr_b = np.zeros(256, dtype=np.uint8)
            pl_b = np.concatenate([hdr_b, p1_b.reshape(-1), pidx_b.view(np.uint8), pbytes_b])
            b2_b = np.concatenate([p0_b.reshape(-1), tbl_b.reshape(-1)])
            pl_bufs_b.append(pl_b); b2_bufs_b.append(b2_b)
            p1s_b.append(p1_b); pidxs_b.append(pidx_b)
        b2_cat_b = np.concatenate(b2_bufs_b)
        mats_b = (GcMat * 3)()
        b2_cur_b = sum(x.nbytes for x in pl_bufs_b)
        pl_cur_b = 0
        for i, (N_c, K_c) in enumerate(shapes):
            mats_b[i] = GcMat(N_c, K_c, b2_cur_b, pl_cur_b, 256, 256 + p1s_b[i].nbytes, 256 + p1s_b[i].nbytes + pidxs_b[i].nbytes, 1)
            b2_cur_b += b2_bufs_b[i].nbytes
            pl_cur_b += pl_bufs_b[i].nbytes

        av_13_b = rng.integers(-50000, 50001, size=(M_c, 5120), dtype=np.int32)
        sv_13_b = rng.uniform(0.5, 1.5, size=(M_c, 160)).astype(np.float32)
        bad_13_b = np.array([0, 0], dtype=np.uint8)
        av_2_b = rng.integers(-50000, 50001, size=(M_c, 2304), dtype=np.int32)
        sv_2_b = rng.uniform(0.5, 1.5, size=(M_c, 72)).astype(np.float32)
        bad_2_b = np.array([1, 0], dtype=np.uint8)

        times_flush = (ctypes.c_double * 3)()
        dll.hv41c_corr_take_times(times_flush)

        h_ptr_a = dll.hv41c_corr_host(0)
        assert h_ptr_a
        dll.hv41c_corr_hs_wait(0)
        ctypes.memmove(h_ptr_a, b2_cat.ctypes.data, b2_cat.nbytes)
        assert dll.hv41c_corr_load_gc(0, 0, ptrs_arr, sizes_arr, b2_cat.nbytes) == 0
        assert dll.hv41c_corr_decode(0, 0, mats, 8) == 0
        assert dll.hv41c_corr_w13(0, 8, av_13.ctypes.data, sv_13.ctypes.data, bad_13.ctypes.data, M_c) == 0

        h_ptr_b = dll.hv41c_corr_host(1)
        assert h_ptr_b
        dll.hv41c_corr_hs_wait(1)
        ctypes.memmove(h_ptr_b, b2_cat_b.ctypes.data, b2_cat_b.nbytes)
        ptrs_arr_b = (ctypes.c_void_p * 3)(pl_bufs_b[0].ctypes.data, pl_bufs_b[1].ctypes.data, pl_bufs_b[2].ctypes.data)
        sizes_arr_b = (ctypes.c_uint64 * 3)(pl_bufs_b[0].nbytes, pl_bufs_b[1].nbytes, pl_bufs_b[2].nbytes)
        assert dll.hv41c_corr_load_gc(1, 1, ptrs_arr_b, sizes_arr_b, b2_cat_b.nbytes) == 0
        assert dll.hv41c_corr_decode(1, 1, mats_b, 9) == 0
        assert dll.hv41c_corr_w13(1, 9, av_13_b.ctypes.data, sv_13_b.ctypes.data, bad_13_b.ctypes.data, M_c) == 0

        assert dll.hv41c_corr_wait13(8) == 0
        assert dll.hv41c_corr_wait13(9) == 0
        yg_ov_a = np.ctypeslib.as_array(dll.hv41c_corr_yg(8), shape=(M_c, 2304)).copy()
        yu_ov_a = np.ctypeslib.as_array(dll.hv41c_corr_yu(8), shape=(M_c, 2304)).copy()
        yg_ov_b = np.ctypeslib.as_array(dll.hv41c_corr_yg(9), shape=(M_c, 2304)).copy()
        yu_ov_b = np.ctypeslib.as_array(dll.hv41c_corr_yu(9), shape=(M_c, 2304)).copy()

        assert dll.hv41c_corr_w2(0, 8, av_2.ctypes.data, sv_2.ctypes.data, bad_2.ctypes.data, M_c) == 0
        assert dll.hv41c_corr_w2(1, 9, av_2_b.ctypes.data, sv_2_b.ctypes.data, bad_2_b.ctypes.data, M_c) == 0
        assert dll.hv41c_corr_wait2(8) == 0
        assert dll.hv41c_corr_wait2(9) == 0
        y2_ov_a = np.ctypeslib.as_array(dll.hv41c_corr_y2(8), shape=(M_c, 5120)).copy()
        y2_ov_b = np.ctypeslib.as_array(dll.hv41c_corr_y2(9), shape=(M_c, 5120)).copy()

        dec_host_b = np.empty(0x11EE000, dtype=np.uint8)
        assert dll.hv41c_corr_read_decoded(1, dec_host_b.ctypes.data) == 0
        c_w1_b = np.ascontiguousarray(dec_host_b[0 : 0x5A0000].reshape(2304, 2560))
        s_w1_b = np.ascontiguousarray(dec_host_b[0x5A0000 : 0x5FA000].reshape(2304, 160))
        c_w2_b = np.ascontiguousarray(dec_host_b[0x5FA000 : 0x5FA000 + 0x5A0000].reshape(5120, 1152))
        s_w2_b = np.ascontiguousarray(dec_host_b[0x5FA000 + 0x5A0000 : 0xBF4000].reshape(5120, 72))
        c_w3_b = np.ascontiguousarray(dec_host_b[0xBF4000 : 0xBF4000 + 0x5A0000].reshape(2304, 2560))
        s_w3_b = np.ascontiguousarray(dec_host_b[0xBF4000 + 0x5A0000 : 0x11EE000].reshape(2304, 160))

        ref_yg_b = _cpu_dot8(c_w1_b, s_w1_b, av_13_b, sv_13_b, bad_13_b, 2304, 5120)
        ref_yu_b = _cpu_dot8(c_w3_b, s_w3_b, av_13_b, sv_13_b, bad_13_b, 2304, 5120)
        ref_y2_b = _cpu_dot8(c_w2_b, s_w2_b, av_2_b, sv_2_b, bad_2_b, 5120, 2304)

        bad_ov = (
            int(np.count_nonzero(yg_ov_a.view(np.uint32) != ref_yg.view(np.uint32)))
            + int(np.count_nonzero(yu_ov_a.view(np.uint32) != ref_yu.view(np.uint32)))
            + int(np.count_nonzero(y2_ov_a.view(np.uint32) != ref_y2.view(np.uint32)))
            + int(np.count_nonzero(yg_ov_b.view(np.uint32) != ref_yg_b.view(np.uint32)))
            + int(np.count_nonzero(yu_ov_b.view(np.uint32) != ref_yu_b.view(np.uint32)))
            + int(np.count_nonzero(y2_ov_b.view(np.uint32) != ref_y2_b.view(np.uint32)))
        )
        times_ov = (ctypes.c_double * 3)()
        dll.hv41c_corr_take_times(times_ov)
        assert times_ov[0] > 0.0 and times_ov[1] > 0.0 and times_ov[2] > 0.0, (
            f"expected positive take_times after async overlap, got {list(times_ov)}"
        )
        n_ov = 2 * n_corr
        print(
            f"  corr_abi async overlap 2-expert (slots 0/8 + 1/9): outputs={n_ov}, "
            f"mismatches={bad_ov}, take_times_ms={[round(float(x), 3) for x in times_ov]}"
        )
        total_out += n_ov
        total_bad += bad_ov

    return total_out, total_bad


if __name__ == "__main__":
    sys.exit(main())
