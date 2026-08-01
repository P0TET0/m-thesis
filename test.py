import ctypes
import time
from ctypes import wintypes


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MAPVK_VK_TO_VSC = 0

VK_SPACE = 0x20
VK_Q = 0x51

PRESS_HOLD_SECONDS = 0.02
INTERVAL_SECONDS = 0.03
TARGET_KEYS = (("SPACE", VK_SPACE),) + tuple(
    (chr(vk_code), vk_code)
    for vk_code in range(ord("A"), ord("Z") + 1)
    if vk_code != VK_Q
)

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else wintypes.DWORD

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class INPUTUNION(ctypes.Union):
    _fields_ = (
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    )


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (
        ("type", wintypes.DWORD),
        ("u", INPUTUNION),
    )


user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
user32.GetAsyncKeyState.restype = wintypes.SHORT
user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
user32.MapVirtualKeyW.restype = wintypes.UINT


def is_q_pressed() -> bool:
    return bool(user32.GetAsyncKeyState(VK_Q) & 0x8000)


def send_scan_code(scan_code: int, key_up: bool = False) -> None:
    flags = KEYEVENTF_SCANCODE
    if key_up:
        flags |= KEYEVENTF_KEYUP

    event = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(
            wVk=0,
            wScan=scan_code,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        ),
    )

    sent = user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
    if sent != 1:
        raise ctypes.WinError(ctypes.get_last_error())


def press_keys(vk_codes: tuple[int, ...]) -> None:
    scan_codes = [user32.MapVirtualKeyW(vk_code, MAPVK_VK_TO_VSC) for vk_code in vk_codes]

    for scan_code in scan_codes:
        send_scan_code(scan_code, key_up=False)

    time.sleep(PRESS_HOLD_SECONDS)

    for scan_code in reversed(scan_codes):
        send_scan_code(scan_code, key_up=True)


def main() -> None:
    key_names = ", ".join(name for name, _ in TARGET_KEYS)
    print(f"Starts pressing {key_names} in 3 seconds. Press q to stop.")
    time.sleep(3)

    try:
        while not is_q_pressed():
            press_keys(tuple(vk_code for _, vk_code in TARGET_KEYS))
            time.sleep(INTERVAL_SECONDS)
    except PermissionError as error:
        print(f"SendInput failed: {error}")
        print("If the target app is running as administrator, run this script as administrator too.")
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopped.")


if __name__ == "__main__":
    main()
