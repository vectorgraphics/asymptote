#!/usr/bin/env python3
"""Check that asy can render a 3D scene to a bitmap.

Exits 0 on success, 1 on failure, and 77 (skipped) if asy was built without
Vulkan or no Vulkan device is available.

Usage:
    python3 tests/test_render3d.py
    python3 tests/test_render3d.py --asy PATH --asy-base-dir PATH
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import subprocess
import sys
import tempfile
from typing import Optional

# One directory up from this file is the asymptote source root.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ASY = os.path.join(SCRIPT_DIR, "asy")
_DEFAULT_BASE_DIR = os.path.join(SCRIPT_DIR, "base")

SKIP = 77

SCENE = """\
import three;
size(2cm);
draw(unitsphere,red);
"""


def asy_has_vulkan(asy: str) -> bool:
    """Return True if `asy --version` lists Vulkan among the enabled options."""
    result = subprocess.run(
        [asy, "--version"], capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    enabled = output.split("ENABLED OPTIONS:", 1)[-1].split("DISABLED OPTIONS:")[0]
    return any(line.split()[:1] == ["Vulkan"] for line in enabled.splitlines())


class _VkInstanceCreateInfo(ctypes.Structure):  # pylint: disable=too-few-public-methods
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
    ]


_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
_VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = 0x1
_PORTABILITY_EXTENSION = b"VK_KHR_portability_enumeration"


def _load_vulkan() -> Optional[ctypes.CDLL]:
    names = ["libvulkan.so.1", "libvulkan.1.dylib", "vulkan-1.dll"]
    found = ctypes.util.find_library("vulkan")
    if found:
        names.append(found)
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def vulkan_device_count() -> int:
    """Return the number of Vulkan physical devices the loader reports."""
    lib = _load_vulkan()
    if lib is None:
        return 0
    lib.vkCreateInstance.restype = ctypes.c_int
    lib.vkCreateInstance.argtypes = [
        ctypes.POINTER(_VkInstanceCreateInfo),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.vkEnumeratePhysicalDevices.restype = ctypes.c_int
    lib.vkEnumeratePhysicalDevices.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    lib.vkDestroyInstance.restype = None
    lib.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    # Ask for portability drivers (MoltenVK) first; older loaders reject the
    # extension, so fall back to a plain instance.
    extensions = (ctypes.c_char_p * 1)(_PORTABILITY_EXTENSION)
    for portability in (True, False):
        if portability:
            info = _VkInstanceCreateInfo(
                sType=_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                flags=_VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
                enabledExtensionCount=1,
                ppEnabledExtensionNames=extensions,
            )
        else:
            info = _VkInstanceCreateInfo(sType=_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
        instance = ctypes.c_void_p()
        if lib.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance)):
            continue
        count = ctypes.c_uint32(0)
        result = lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
        lib.vkDestroyInstance(instance, None)
        return count.value if result == 0 else 0
    return 0


def render(asy: str, base_dir: str) -> tuple[bool, str]:
    """Render SCENE to PNG; return (success, diagnostic)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = os.path.join(tmpdir, "render3d.asy")
        output = os.path.join(tmpdir, "render3d.png")
        with open(source, "w", encoding="utf-8") as f:
            f.write(SCENE)
        try:
            result = subprocess.run(
                [
                    asy,
                    "-noV",
                    "-sysdir",
                    base_dir,
                    "-f",
                    "png",
                    "-render=1",
                    source,
                ],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return False, "asy timed out"
        log = result.stdout + result.stderr
        if result.returncode != 0:
            return False, f"asy exited with status {result.returncode}\n{log}"
        if not os.path.isfile(output):
            return False, f"asy exited with status 0 but wrote no image\n{log}"
        with open(output, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return False, f"{output} is not a PNG file\n{log}"
        return True, log


def main() -> int:
    parser = argparse.ArgumentParser(description="3D bitmap rendering test.")
    parser.add_argument(
        "--asy",
        default=_DEFAULT_ASY,
        help="path to the asy executable (default: %(default)s)",
    )
    parser.add_argument(
        "--asy-base-dir",
        default=_DEFAULT_BASE_DIR,
        help="path to the asy base/sysdir (default: %(default)s)",
    )
    args = parser.parse_args()

    print("Testing 3D rendering ... ", end="", flush=True)
    if not asy_has_vulkan(args.asy):
        print("SKIPPED (asy was built without Vulkan)")
        return SKIP
    if vulkan_device_count() == 0:
        print("SKIPPED (no Vulkan device found)")
        return SKIP

    success, diagnostic = render(args.asy, args.asy_base_dir)
    if not success:
        print("FAILED")
        print(diagnostic)
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
