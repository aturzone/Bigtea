//! The Win32 surface this app uses, declared by hand.
//!
//! Every item here is something Windows already ships in `user32`, `gdi32` or
//! `kernel32`. Declaring them is mechanical; taking a GUI crate instead would
//! add more code to the dependency graph than the whole rest of this workspace
//! contains, and the reason a Chaos binary starts on a machine with no runtime
//! installed is that it links almost nothing.
//!
//! Only what is called appears below. An unused `extern` declaration is a
//! promise about a symbol's signature that nothing ever checks.

// The names are Windows' own. `HWND` renamed to `Hwnd` would read better and
// would also stop these declarations matching the documentation they are
// checked against, which is the only review this file can get.
#![allow(non_snake_case, non_camel_case_types, clippy::upper_case_acronyms)]

use std::ffi::c_void;

pub type HWND = *mut c_void;
pub type HINSTANCE = *mut c_void;
pub type HMENU = *mut c_void;
pub type HDC = *mut c_void;
pub type HBRUSH = *mut c_void;
pub type HFONT = *mut c_void;
pub type HGDIOBJ = *mut c_void;
pub type HICON = *mut c_void;
pub type HCURSOR = *mut c_void;
pub type WPARAM = usize;
pub type LPARAM = isize;
pub type LRESULT = isize;
pub type COLORREF = u32;
pub type BOOL = i32;

pub type WNDPROC = unsafe extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT;

#[repr(C)]
pub struct WNDCLASSW {
    pub style: u32,
    pub lpfnWndProc: Option<WNDPROC>,
    pub cbClsExtra: i32,
    pub cbWndExtra: i32,
    pub hInstance: HINSTANCE,
    pub hIcon: HICON,
    pub hCursor: HCURSOR,
    pub hbrBackground: HBRUSH,
    pub lpszMenuName: *const u16,
    pub lpszClassName: *const u16,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct POINT {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RECT {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[repr(C)]
pub struct MSG {
    pub hwnd: HWND,
    pub message: u32,
    pub wParam: WPARAM,
    pub lParam: LPARAM,
    pub time: u32,
    pub pt: POINT,
}

impl Default for MSG {
    fn default() -> Self {
        // HWND is a raw pointer, so it has no Default; everything else is zero.
        Self {
            hwnd: std::ptr::null_mut(),
            message: 0,
            wParam: 0,
            lParam: 0,
            time: 0,
            pt: POINT::default(),
        }
    }
}

#[repr(C)]
pub struct PAINTSTRUCT {
    pub hdc: HDC,
    pub fErase: BOOL,
    pub rcPaint: RECT,
    pub fRestore: BOOL,
    pub fIncUpdate: BOOL,
    pub rgbReserved: [u8; 32],
}

#[repr(C)]
pub struct BITMAPINFOHEADER {
    pub biSize: u32,
    pub biWidth: i32,
    pub biHeight: i32,
    pub biPlanes: u16,
    pub biBitCount: u16,
    pub biCompression: u32,
    pub biSizeImage: u32,
    pub biXPelsPerMeter: i32,
    pub biYPelsPerMeter: i32,
    pub biClrUsed: u32,
    pub biClrImportant: u32,
}

// -- window styles ----------------------------------------------------------

pub const WS_OVERLAPPEDWINDOW: u32 = 0x00CF_0000;
pub const WS_CHILD: u32 = 0x4000_0000;
pub const WS_VISIBLE: u32 = 0x1000_0000;
pub const WS_VSCROLL: u32 = 0x0020_0000;
pub const WS_BORDER: u32 = 0x0080_0000;
pub const WS_TABSTOP: u32 = 0x0001_0000;

pub const ES_MULTILINE: u32 = 0x0004;
pub const ES_READONLY: u32 = 0x0800;
pub const ES_AUTOVSCROLL: u32 = 0x0040;
pub const ES_WANTRETURN: u32 = 0x1000;

pub const LBS_NOTIFY: u32 = 0x0001;
/// Owner-draw, because Windows paints a push button from the *theme* and
/// ignores `WM_CTLCOLORBTN` entirely. Without this the buttons come up in the
/// system's grey no matter what the parent says, which on a two-value design is
/// the design gone.
pub const BS_OWNERDRAW: u32 = 0x000B;
/// Same problem in the list: the selection bar is the system highlight colour,
/// which is blue. Owner-draw, keeping the strings so `LB_ADDSTRING` still works.
pub const LBS_OWNERDRAWFIXED: u32 = 0x0010;
pub const LBS_HASSTRINGS: u32 = 0x0040;

pub const SW_SHOW: i32 = 5;

// -- messages ---------------------------------------------------------------

pub const WM_DESTROY: u32 = 0x0002;
pub const WM_SIZE: u32 = 0x0005;
pub const WM_PAINT: u32 = 0x000F;
pub const WM_CLOSE: u32 = 0x0010;
pub const WM_COMMAND: u32 = 0x0111;
pub const WM_CTLCOLOREDIT: u32 = 0x0133;
pub const WM_CTLCOLORLISTBOX: u32 = 0x0134;
pub const WM_CTLCOLORBTN: u32 = 0x0135;
pub const WM_CTLCOLORSTATIC: u32 = 0x0138;
pub const WM_SETFONT: u32 = 0x0030;
/// Our own: the worker thread has produced output and the UI should read it.
pub const WM_APP_TICK: u32 = 0x8000 + 1;

pub const WM_DRAWITEM: u32 = 0x002B;
pub const ODS_SELECTED: u32 = 0x0001;
pub const ODS_DISABLED: u32 = 0x0004;
pub const ODT_LISTBOX: u32 = 2;

#[repr(C)]
pub struct DRAWITEMSTRUCT {
    pub CtlType: u32,
    pub CtlID: u32,
    pub itemID: u32,
    pub itemAction: u32,
    pub itemState: u32,
    pub hwndItem: HWND,
    pub hDC: HDC,
    pub rcItem: RECT,
    pub itemData: usize,
}

pub const LB_SETITEMHEIGHT: u32 = 0x01A0;
pub const LB_GETTEXT: u32 = 0x0189;
pub const LB_GETTEXTLEN: u32 = 0x018A;
pub const EM_SETSEL: u32 = 0x00B1;
pub const EM_REPLACESEL: u32 = 0x00C2;
pub const EM_SCROLLCARET: u32 = 0x00B7;
pub const LB_ADDSTRING: u32 = 0x0180;
pub const LB_GETCURSEL: u32 = 0x0188;
pub const LB_RESETCONTENT: u32 = 0x0184;
pub const LB_SETCURSEL: u32 = 0x0186;
pub const LBN_SELCHANGE: u16 = 1;
pub const BN_CLICKED: u16 = 0;

pub const IDC_ARROW: u32 = 32512;
pub const SRCCOPY: u32 = 0x00CC_0020;
pub const DIB_RGB_COLORS: u32 = 0;
pub const BI_RGB: u32 = 0;
pub const TRANSPARENT: i32 = 1;

pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
/// The child must outlive the parent that spawned it -- used by the
/// uninstaller, which cannot delete the directory it is running from.
pub const DETACHED_PROCESS: u32 = 0x0000_0008;

#[link(name = "user32")]
extern "system" {
    pub fn RegisterClassW(lpWndClass: *const WNDCLASSW) -> u16;
    pub fn CreateWindowExW(
        dwExStyle: u32,
        lpClassName: *const u16,
        lpWindowName: *const u16,
        dwStyle: u32,
        x: i32,
        y: i32,
        nWidth: i32,
        nHeight: i32,
        hWndParent: HWND,
        hMenu: HMENU,
        hInstance: HINSTANCE,
        lpParam: *mut c_void,
    ) -> HWND;
    pub fn DefWindowProcW(hWnd: HWND, Msg: u32, wParam: WPARAM, lParam: LPARAM) -> LRESULT;
    pub fn ShowWindow(hWnd: HWND, nCmdShow: i32) -> BOOL;
    pub fn UpdateWindow(hWnd: HWND) -> BOOL;
    pub fn GetMessageW(lpMsg: *mut MSG, hWnd: HWND, wMsgFilterMin: u32, wMsgFilterMax: u32)
        -> BOOL;
    pub fn TranslateMessage(lpMsg: *const MSG) -> BOOL;
    pub fn DispatchMessageW(lpMsg: *const MSG) -> LRESULT;
    pub fn PostQuitMessage(nExitCode: i32);
    pub fn PostMessageW(hWnd: HWND, Msg: u32, wParam: WPARAM, lParam: LPARAM) -> BOOL;
    pub fn SendMessageW(hWnd: HWND, Msg: u32, wParam: WPARAM, lParam: LPARAM) -> LRESULT;
    pub fn LoadCursorW(hInstance: HINSTANCE, lpCursorName: *const u16) -> HCURSOR;
    pub fn BeginPaint(hWnd: HWND, lpPaint: *mut PAINTSTRUCT) -> HDC;
    pub fn EndPaint(hWnd: HWND, lpPaint: *const PAINTSTRUCT) -> BOOL;
    pub fn FillRect(hDC: HDC, lprc: *const RECT, hbr: HBRUSH) -> i32;
    pub fn GetClientRect(hWnd: HWND, lpRect: *mut RECT) -> BOOL;
    pub fn MoveWindow(
        hWnd: HWND,
        X: i32,
        Y: i32,
        nWidth: i32,
        nHeight: i32,
        bRepaint: BOOL,
    ) -> BOOL;
    pub fn InvalidateRect(hWnd: HWND, lpRect: *const RECT, bErase: BOOL) -> BOOL;
    pub fn DestroyWindow(hWnd: HWND) -> BOOL;
    pub fn SetWindowTextW(hWnd: HWND, lpString: *const u16) -> BOOL;
    pub fn EnableWindow(hWnd: HWND, bEnable: BOOL) -> BOOL;
    pub fn GetDlgItem(hDlg: HWND, nIDDlgItem: i32) -> HWND;
    pub fn GetWindowTextW(hWnd: HWND, lpString: *mut u16, nMaxCount: i32) -> i32;
    pub fn GetWindowTextLengthW(hWnd: HWND) -> i32;
    pub fn GetWindowRect(hWnd: HWND, lpRect: *mut RECT) -> BOOL;
    pub fn ScreenToClient(hWnd: HWND, lpPoint: *mut POINT) -> BOOL;
}

#[link(name = "gdi32")]
extern "system" {
    pub fn CreateSolidBrush(color: COLORREF) -> HBRUSH;
    pub fn DeleteObject(ho: HGDIOBJ) -> BOOL;
    pub fn SelectObject(hdc: HDC, h: HGDIOBJ) -> HGDIOBJ;
    pub fn SetTextColor(hdc: HDC, color: COLORREF) -> COLORREF;
    pub fn SetBkColor(hdc: HDC, color: COLORREF) -> COLORREF;
    pub fn SetBkMode(hdc: HDC, mode: i32) -> i32;
    pub fn TextOutW(hdc: HDC, x: i32, y: i32, lpString: *const u16, c: i32) -> BOOL;
    pub fn CreateFontW(
        cHeight: i32,
        cWidth: i32,
        cEscapement: i32,
        cOrientation: i32,
        cWeight: i32,
        bItalic: u32,
        bUnderline: u32,
        bStrikeOut: u32,
        iCharSet: u32,
        iOutPrecision: u32,
        iClipPrecision: u32,
        iQuality: u32,
        iPitchAndFamily: u32,
        pszFaceName: *const u16,
    ) -> HFONT;
    pub fn StretchDIBits(
        hdc: HDC,
        xDest: i32,
        yDest: i32,
        DestWidth: i32,
        DestHeight: i32,
        xSrc: i32,
        ySrc: i32,
        SrcWidth: i32,
        SrcHeight: i32,
        lpBits: *const c_void,
        lpbmi: *const BITMAPINFOHEADER,
        iUsage: u32,
        rop: u32,
    ) -> i32;
    pub fn Rectangle(hdc: HDC, left: i32, top: i32, right: i32, bottom: i32) -> BOOL;
    pub fn CreatePen(iStyle: i32, cWidth: i32, color: COLORREF) -> HGDIOBJ;
}

#[link(name = "kernel32")]
extern "system" {
    pub fn GetModuleHandleW(lpModuleName: *const u16) -> HINSTANCE;
}

/// The title bar is drawn by the desktop compositor, not by us, so it stays
/// light however the client area is painted. `DWMWA_USE_IMMERSIVE_DARK_MODE`
/// is the only way to make it match, and it is a no-op on builds too old to
/// know the attribute -- which is why the return value is ignored.
pub const DWMWA_USE_IMMERSIVE_DARK_MODE: u32 = 20;

#[link(name = "dwmapi")]
extern "system" {
    pub fn DwmSetWindowAttribute(
        hwnd: HWND,
        dwAttribute: u32,
        pvAttribute: *const c_void,
        cbAttribute: u32,
    ) -> i32;
}

/// A NUL-terminated UTF-16 buffer, which is what every `...W` entry point wants.
pub fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// `RGB()` from `wingdi.h`: **0x00bbggrr**, not the 0xrrggbb everyone expects.
/// Reversing it silently swaps red and blue, which on a two-colour design is
/// invisible -- black and white are palindromes in this encoding, so a mistake
/// here only shows up the moment a third colour is added.
pub const fn rgb(r: u8, g: u8, b: u8) -> COLORREF {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16)
}

pub const BLACK: COLORREF = rgb(0, 0, 0);
pub const WHITE: COLORREF = rgb(255, 255, 255);

// -- the registry, and telling the world PATH changed ------------------------
//
// Used by the installer. A per-user install writes exactly two places: its own
// folder, and `HKEY_CURRENT_USER`. Nothing here needs administrator rights.

pub type HKEY = *mut c_void;
pub const HKEY_CURRENT_USER: HKEY = 0x8000_0001u32 as usize as HKEY;
pub const KEY_READ: u32 = 0x0002_0019;
pub const KEY_WRITE: u32 = 0x0002_0006;
pub const REG_SZ: u32 = 1;
pub const HWND_BROADCAST: HWND = 0xFFFF_usize as HWND;
pub const WM_SETTINGCHANGE: u32 = 0x001A;
pub const SMTO_ABORTIFHUNG: u32 = 0x0002;

#[link(name = "advapi32")]
extern "system" {
    fn RegOpenKeyExW(k: HKEY, sub: *const u16, opt: u32, sam: u32, out: *mut HKEY) -> i32;
    fn RegCreateKeyExW(
        k: HKEY,
        sub: *const u16,
        reserved: u32,
        class: *const u16,
        options: u32,
        sam: u32,
        sa: *const c_void,
        out: *mut HKEY,
        disp: *mut u32,
    ) -> i32;
    fn RegQueryValueExW(
        k: HKEY,
        name: *const u16,
        reserved: *const u32,
        ty: *mut u32,
        data: *mut u8,
        len: *mut u32,
    ) -> i32;
    fn RegSetValueExW(
        k: HKEY,
        name: *const u16,
        reserved: u32,
        ty: u32,
        data: *const u8,
        len: u32,
    ) -> i32;
    fn RegCloseKey(k: HKEY) -> i32;
    fn RegDeleteTreeW(k: HKEY, sub: *const u16) -> i32;
}

#[link(name = "user32")]
extern "system" {
    pub fn SendMessageTimeoutW(
        hWnd: HWND,
        Msg: u32,
        wParam: WPARAM,
        lParam: LPARAM,
        flags: u32,
        timeout: u32,
        result: *mut usize,
    ) -> isize;
}

/// Read a string value, or `None` if the key or value is absent.
/// Reads only from `HKEY_CURRENT_USER`, and takes no hive parameter.
///
/// An `HKEY` is a raw pointer, so accepting one would make this a safe function
/// that dereferences whatever it is handed. A per-user install writes exactly
/// one hive, so the hive is not a parameter and the signature has no pointer in
/// it at all.
pub fn hkcu_read_string(sub: &str, name: &str) -> Option<String> {
    let root = HKEY_CURRENT_USER;
    unsafe {
        let mut k: HKEY = std::ptr::null_mut();
        if RegOpenKeyExW(root, wide(sub).as_ptr(), 0, KEY_READ, &mut k) != 0 {
            return None;
        }
        let n = wide(name);
        let mut len: u32 = 0;
        // Ask for the size first: a PATH can be far longer than any buffer
        // guessed up front, and truncating it here would silently destroy it
        // on the next write.
        let rc = RegQueryValueExW(
            k,
            n.as_ptr(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut len,
        );
        if rc != 0 || len == 0 {
            RegCloseKey(k);
            return None;
        }
        let mut buf = vec![0u8; len as usize];
        let rc = RegQueryValueExW(
            k,
            n.as_ptr(),
            std::ptr::null(),
            std::ptr::null_mut(),
            buf.as_mut_ptr(),
            &mut len,
        );
        RegCloseKey(k);
        if rc != 0 {
            return None;
        }
        let u16s: Vec<u16> = buf
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .take_while(|&c| c != 0)
            .collect();
        Some(String::from_utf16_lossy(&u16s))
    }
}

/// Write a string value, creating the key if needed.
pub fn hkcu_write_string(sub: &str, name: &str, value: &str) -> bool {
    let root = HKEY_CURRENT_USER;
    unsafe {
        let mut k: HKEY = std::ptr::null_mut();
        let mut disp: u32 = 0;
        if RegCreateKeyExW(
            root,
            wide(sub).as_ptr(),
            0,
            std::ptr::null(),
            0,
            KEY_WRITE,
            std::ptr::null(),
            &mut k,
            &mut disp,
        ) != 0
        {
            return false;
        }
        let v = wide(value);
        let bytes: Vec<u8> = v.iter().flat_map(|c| c.to_le_bytes()).collect();
        let rc = RegSetValueExW(
            k,
            wide(name).as_ptr(),
            0,
            REG_SZ,
            bytes.as_ptr(),
            bytes.len() as u32,
        );
        RegCloseKey(k);
        rc == 0
    }
}

pub fn hkcu_delete_key(sub: &str) -> bool {
    let root = HKEY_CURRENT_USER;
    unsafe { RegDeleteTreeW(root, wide(sub).as_ptr()) == 0 }
}
