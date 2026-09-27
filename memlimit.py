"""
memlimit.py  -  keeps the model scripts under a RAM limit, so a big run can't freeze the PC.

If a script goes over the limit, Windows refuses to give it more memory: that script stops
with a "MemoryError" / "out of memory" message, and the rest of the computer keeps working.

The limit is 7 GB by default. Change it with:
    python run_pipeline.py --max-ram-gb 6         (or update_model.bat --max-ram-gb 6)
or for a single script, set it first in PowerShell:
    $env:LOL_MAX_RAM_GB = "6"
0 turns the limit off.
"""

import os
import sys

DEFAULT_GB = 7.0
_ENV = "LOL_MAX_RAM_GB"
_SET_BY_PARENT = "LOL_RAM_LIMIT_ACTIVE"
_job = None          # Windows job handle, kept open for the life of the process


def _windows_job(limit_bytes):
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JobObjectExtendedLimitInformation = 9
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                            wintypes.DWORD]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.QueryInformationJobObject.restype = wintypes.BOOL
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                              wintypes.DWORD, ctypes.c_void_p]

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY
    info.JobMemoryLimit = limit_bytes
    if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                       ctypes.byref(info), ctypes.sizeof(info)):
        return None
    # every program this process starts afterwards joins the same job (shares the limit)
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        return None

    def peak():
        q = EXTENDED_LIMIT()
        ok = k32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation,
                                           ctypes.byref(q), ctypes.sizeof(q), None)
        return q.PeakJobMemoryUsed if ok else None
    return peak


def apply(gb=None, quiet=False):
    """Limit this process (and anything it starts) to `gb` GB of RAM. Returns the limit
    in GB, or None if no limit was set."""
    global _job
    if os.environ.get(_SET_BY_PARENT):          # run_pipeline already limits us
        return float(os.environ.get(_ENV, DEFAULT_GB))
    try:
        gb = float(gb if gb is not None else os.environ.get(_ENV, DEFAULT_GB))
    except ValueError:
        gb = DEFAULT_GB
    if gb <= 0:
        return None
    limit = int(gb * 1024 ** 3)
    ok = False
    try:
        if sys.platform == "win32":
            _job = _windows_job(limit)
            ok = _job is not None
        elif sys.platform.startswith("linux"):
            import resource
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            new = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
            resource.setrlimit(resource.RLIMIT_AS, (new, hard))
            ok = True
    except Exception:
        ok = False
    if ok:
        os.environ[_ENV] = str(gb)
        os.environ[_SET_BY_PARENT] = "1"
        if not quiet:
            print(f"RAM limit: {gb:g} GB (change with --max-ram-gb, 0 = no limit)", flush=True)
        return gb
    if not quiet:
        print("NOTE: couldn't set a RAM limit on this system - running without one.", flush=True)
    return None


def peak_gb():
    """Highest RAM use so far of everything under the limit (Windows only), in GB."""
    if _job is None:
        return None
    try:
        v = _job()
        return v / 1024 ** 3 if v else None
    except Exception:
        return None


def process_gb():
    """(current, peak) RAM of THIS process in GB, or (None, None) if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            c = PMC()
            c.cb = ctypes.sizeof(c)
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return c.PagefileUsage / 1024 ** 3, c.PeakPagefileUsage / 1024 ** 3
        elif sys.platform.startswith("linux"):
            cur = peak = None
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        cur = int(line.split()[1]) / 1024 ** 2
                    elif line.startswith("VmHWM:"):
                        peak = int(line.split()[1]) / 1024 ** 2
            return cur, peak
    except Exception:
        pass
    return None, None


def ram_note():
    """' [RAM 1.2 GB, peak 2.3 GB]' for progress messages ('' if unknown)."""
    cur, peak = process_gb()
    return f"  [RAM {cur:.1f} GB, peak {peak:.1f} GB]" if cur is not None else ""


def is_memory_error(text):
    t = text.lower()
    return any(s in t for s in ("memoryerror", "bad_alloc", "unable to allocate",
                                "out of memory", "cannot allocate memory"))
