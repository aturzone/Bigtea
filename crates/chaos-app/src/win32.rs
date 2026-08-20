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
/// **The cure for a window that flickers once a second.**
///
/// Without it, the parent's `WM_PAINT` paints the whole client area *including*
/// the rectangles its child controls occupy, and the children then repaint on
/// top. The transcript, the list and every box flashed on every tick, which is
/// exactly the "non-stop glitch" Atur reported -- and double-buffering the
/// parent cannot fix it, because the flicker is the children, not the parent.
pub const WS_CLIPCHILDREN: u32 = 0x0200_0000;
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
/// Hide a control without destroying it. `show_page` reveals one page's
/// controls and hides the rest; nothing is ever torn down.
pub const SW_HIDE: i32 = 0;

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
/// The control has the caret. Owner-draw means Windows draws no focus ring,
/// so keyboard users would otherwise have no idea where they are.
pub const ODS_FOCUS: u32 = 0x0010;
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
/// A read-only EDIT **silently ignores `EM_REPLACESEL`**. It returns nothing,
/// sets no error, and the text simply does not appear -- which is why the
/// transcript stayed empty while the model was answering normally. Clear the
/// flag, append, set it again.
pub const EM_SETREADONLY: u32 = 0x00CF;
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

/// `WM_SETICON`, and the two sizes it takes: the small one is the title bar and
/// the Alt-Tab strip, the big one is the taskbar. Setting only one leaves the
/// other as the default blank page.
pub const WM_SETICON: u32 = 0x0080;
pub const ICON_SMALL: WPARAM = 0;
pub const ICON_BIG: WPARAM = 1;
pub const IMAGE_ICON: u32 = 1;
pub const LR_DEFAULTSIZE: u32 = 0x0000_0040;
pub const LR_SHARED: u32 = 0x0000_8000;

pub const MB_YESNO: u32 = 0x0000_0004;
pub const MB_ICONWARNING: u32 = 0x0000_0030;
pub const IDYES: i32 = 6;

pub const MB_ICONERROR: u32 = 0x0000_0010;
pub const MB_OK: u32 = 0x0000_0000;
pub const MB_ICONINFORMATION: u32 = 0x0000_0040;
pub const MB_ICONQUESTION: u32 = 0x0000_0020;

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
    /// Used only by the panic hook: with `panic = "abort"` and no console,
    /// a message box is the one place a crash can still say something.
    pub fn MessageBoxW(hWnd: HWND, text: *const u16, caption: *const u16, uType: u32) -> i32;
    pub fn LoadIconW(hInstance: HINSTANCE, lpIconName: *const u16) -> HICON;
    pub fn LoadImageW(
        hInst: HINSTANCE,
        name: *const u16,
        ty: u32,
        cx: i32,
        cy: i32,
        fuLoad: u32,
    ) -> *mut c_void;
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

// -- menus -------------------------------------------------------------------
//
// A menu bar is the one place a Windows application is expected to list
// everything it can do. The old window had none, so every capability had to be
// a button, and the buttons had to share one column.

pub const MF_STRING: u32 = 0x0000;
pub const MF_POPUP: u32 = 0x0010;
pub const MF_SEPARATOR: u32 = 0x0800;
pub const MF_CHECKED: u32 = 0x0008;
pub const MF_UNCHECKED: u32 = 0x0000;
pub const MF_BYCOMMAND: u32 = 0x0000;
pub const MF_ENABLED: u32 = 0x0000;
pub const MF_GRAYED: u32 = 0x0001;

/// One accelerator. `fVirt` carries `FVIRTKEY` plus modifiers; `key` is a
/// virtual-key code, not a character, whenever `FVIRTKEY` is set.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ACCEL {
    pub fVirt: u8,
    pub key: u16,
    pub cmd: u16,
}

pub const FVIRTKEY: u8 = 0x01;
pub const FCONTROL: u8 = 0x08;
pub type HACCEL = *mut c_void;

pub const VK_ESCAPE: u16 = 0x1B;
pub const VK_RETURN: u16 = 0x0D;
pub const VK_F5: u16 = 0x74;

#[link(name = "user32")]
extern "system" {
    pub fn CreateMenu() -> HMENU;
    pub fn CreatePopupMenu() -> HMENU;
    pub fn AppendMenuW(hMenu: HMENU, uFlags: u32, uIDNewItem: usize, lpNewItem: *const u16)
        -> BOOL;
    pub fn SetMenu(hWnd: HWND, hMenu: HMENU) -> BOOL;
    pub fn DrawMenuBar(hWnd: HWND) -> BOOL;
    pub fn CheckMenuRadioItem(hmenu: HMENU, first: u32, last: u32, check: u32, flags: u32) -> BOOL;
    pub fn EnableMenuItem(hMenu: HMENU, uIDEnableItem: u32, uEnable: u32) -> BOOL;
    /// A tick beside one item, for a setting that is on or off. Distinct from
    /// `CheckMenuRadioItem`, which is one-of-many.
    pub fn CheckMenuItem(hMenu: HMENU, uIDCheckItem: u32, uCheck: u32) -> u32;
    pub fn GetMenu(hWnd: HWND) -> HMENU;
    pub fn CreateAcceleratorTableW(paccel: *const ACCEL, cAccel: i32) -> HACCEL;
    pub fn TranslateAcceleratorW(hWnd: HWND, hAccTable: HACCEL, lpMsg: *mut MSG) -> i32;
    pub fn SetFocus(hWnd: HWND) -> HWND;
    pub fn SetTimer(hWnd: HWND, nIDEvent: usize, uElapse: u32, lpTimerFunc: usize) -> usize;
    pub fn KillTimer(hWnd: HWND, uIDEvent: usize) -> BOOL;
    pub fn TrackMouseEvent(lpEventTrack: *mut TRACKMOUSEEVENT) -> BOOL;
    pub fn OpenClipboard(hWndNewOwner: HWND) -> BOOL;
    pub fn EmptyClipboard() -> BOOL;
    pub fn SetClipboardData(uFormat: u32, hMem: *mut c_void) -> *mut c_void;
    pub fn CloseClipboard() -> BOOL;
    pub fn DrawTextW(
        hdc: HDC,
        lpchText: *const u16,
        cchText: i32,
        lprc: *mut RECT,
        format: u32,
    ) -> i32;
}

pub const WM_TIMER: u32 = 0x0113;
pub const WM_MOUSEMOVE: u32 = 0x0200;
pub const WM_MOUSELEAVE: u32 = 0x02A3;
pub const WM_ERASEBKGND: u32 = 0x0014;
pub const WM_GETMINMAXINFO: u32 = 0x0024;
pub const WM_KEYDOWN: u32 = 0x0100;
pub const WM_SETCURSOR: u32 = 0x0020;
pub const WM_INITMENUPOPUP: u32 = 0x0117;

/// Clipboard format for UTF-16 text, which is the only one worth writing.
pub const CF_UNICODETEXT: u32 = 13;
pub const GMEM_MOVEABLE: u32 = 0x0002;

#[repr(C)]
pub struct TRACKMOUSEEVENT {
    pub cbSize: u32,
    pub dwFlags: u32,
    pub hwndTrack: HWND,
    pub dwHoverTime: u32,
}

pub const TME_LEAVE: u32 = 0x0000_0002;

#[repr(C)]
pub struct MINMAXINFO {
    pub ptReserved: POINT,
    pub ptMaxSize: POINT,
    pub ptMaxPosition: POINT,
    pub ptMinTrackSize: POINT,
    pub ptMaxTrackSize: POINT,
}

// -- text ---------------------------------------------------------------------

/// `DrawTextW` flags. `TextOutW` cannot align, wrap or ellipsise, which is why
/// the old button labels were centred by guessing seven pixels a character.
pub const DT_LEFT: u32 = 0x0000;
pub const DT_CENTER: u32 = 0x0001;
pub const DT_RIGHT: u32 = 0x0002;
pub const DT_VCENTER: u32 = 0x0004;
pub const DT_SINGLELINE: u32 = 0x0020;
pub const DT_WORDBREAK: u32 = 0x0010;
pub const DT_END_ELLIPSIS: u32 = 0x0000_8000;

/// `EN_CHANGE`: an `EDIT` reporting that its text is now different, which is
/// how a window notices the user typed a different path.
pub const EN_CHANGE: u16 = 0x0300;
pub const DT_CALCRECT: u32 = 0x0400;
pub const DT_NOPREFIX: u32 = 0x0800;

#[repr(C)]
#[derive(Default)]
pub struct SIZE {
    pub cx: i32,
    pub cy: i32,
}

#[link(name = "gdi32")]
extern "system" {
    pub fn GetTextExtentPoint32W(hdc: HDC, lpString: *const u16, c: i32, psizl: *mut SIZE) -> BOOL;
    pub fn CreateCompatibleDC(hdc: HDC) -> HDC;
    pub fn CreateCompatibleBitmap(hdc: HDC, cx: i32, cy: i32) -> HGDIOBJ;
    pub fn BitBlt(
        hdc: HDC,
        x: i32,
        y: i32,
        cx: i32,
        cy: i32,
        hdcSrc: HDC,
        x1: i32,
        y1: i32,
        rop: u32,
    ) -> BOOL;
    pub fn DeleteDC(hdc: HDC) -> BOOL;
    pub fn MoveToEx(hdc: HDC, x: i32, y: i32, lppt: *mut POINT) -> BOOL;
    pub fn LineTo(hdc: HDC, x: i32, y: i32) -> BOOL;
}

pub type HANDLE = *mut c_void;

/// Wait on another process. `SYNCHRONIZE` is the whole access right needed --
/// asking for more fails on a process you did not create.
pub const SYNCHRONIZE: u32 = 0x0010_0000;
pub const WAIT_OBJECT_0: u32 = 0;

/// Enough to ask another process how much memory it is using, and no more.
///
/// `PROCESS_QUERY_LIMITED_INFORMATION` rather than `PROCESS_QUERY_INFORMATION`:
/// it is the narrower right, it is what a working-set query needs, and it is
/// granted in cases the wider one is not.
pub const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
pub const PROCESS_VM_READ: u32 = 0x0010;

/// What `GetProcessMemoryInfo` fills in.
///
/// Declared in full because the call validates `cb` against its own idea of the
/// size, and a short struct is a silent failure rather than an error.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct PROCESS_MEMORY_COUNTERS {
    pub cb: u32,
    pub PageFaultCount: u32,
    pub PeakWorkingSetSize: usize,
    pub WorkingSetSize: usize,
    pub QuotaPeakPagedPoolUsage: usize,
    pub QuotaPagedPoolUsage: usize,
    pub QuotaPeakNonPagedPoolUsage: usize,
    pub QuotaNonPagedPoolUsage: usize,
    pub PagefileUsage: usize,
    pub PeakPagefileUsage: usize,
}

#[link(name = "kernel32")]
extern "system" {
    pub fn GetCurrentProcessId() -> u32;
    pub fn OpenProcess(dwDesiredAccess: u32, bInheritHandle: BOOL, dwProcessId: u32) -> HANDLE;
    pub fn WaitForSingleObject(hHandle: HANDLE, dwMilliseconds: u32) -> u32;
    pub fn CloseHandle(hObject: HANDLE) -> BOOL;
    /// **In kernel32, not psapi.** Since Windows 7 the psapi entry points are
    /// forwarders; linking `psapi` as well works and is one more dependency for
    /// the installer to be wrong about.
    pub fn K32GetProcessMemoryInfo(
        Process: HANDLE,
        ppsmemCounters: *mut PROCESS_MEMORY_COUNTERS,
        cb: u32,
    ) -> BOOL;
    pub fn LoadLibraryW(lpLibFileName: *const u16) -> HANDLE;
    pub fn GetProcAddress(hModule: HANDLE, lpProcName: *const i8) -> *mut c_void;
}

#[link(name = "user32")]
extern "system" {
    pub fn SystemParametersInfoW(
        uiAction: u32,
        uiParam: u32,
        pvParam: *mut c_void,
        fWinIni: u32,
    ) -> BOOL;
    pub fn SetProcessDPIAware() -> BOOL;
}

/// What `SHBrowseForFolderW` is given.
#[repr(C)]
pub struct BROWSEINFOW {
    pub hwndOwner: HWND,
    pub pidlRoot: *mut c_void,
    pub pszDisplayName: *mut u16,
    pub lpszTitle: *const u16,
    pub ulFlags: u32,
    pub lpfn: Option<unsafe extern "system" fn(HWND, u32, isize, isize) -> i32>,
    pub lParam: isize,
    pub iImage: i32,
}

#[link(name = "shell32")]
extern "system" {
    pub fn SHBrowseForFolderW(lpbi: *mut BROWSEINFOW) -> *mut c_void;
    pub fn SHGetPathFromIDListW(pidl: *mut c_void, pszPath: *mut u16) -> BOOL;
}

#[link(name = "ole32")]
extern "system" {
    pub fn CoInitializeEx(pvReserved: *mut c_void, dwCoInit: u32) -> i32;
    pub fn CoTaskMemFree(pv: *mut c_void);
}

/// Ask the user for a folder, returning what they picked.
///
/// # Why a dialog and not a text box
///
/// The models folder was a field you typed a path into. A typo there is silent:
/// the list simply comes up empty, and nothing says the path does not exist.
/// A picked folder cannot be misspelled, and it starts from where the setting
/// already points, so "change it slightly" is a click rather than a retype.
///
/// `SHBrowseForFolderW` rather than the newer `IFileOpenDialog`: one call, no
/// COM interface pointers to release by hand, present on every Windows this
/// ships to.
pub fn pick_folder(owner: HWND, title: &str, start: Option<&str>) -> Option<String> {
    // BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE — a real folder, resizable.
    const FLAGS: u32 = 0x0001 | 0x0040;
    let title_w: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
    let start_w: Option<Vec<u16>> = start
        .filter(|p| !p.is_empty())
        .map(|p| p.encode_utf16().chain(std::iter::once(0)).collect());

    // SAFETY: every pointer is to a local that outlives the call, and the
    // callback only preselects a starting folder.
    unsafe {
        // COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE. Already-initialised
        // returns a failure code that is not one, which is why it is ignored.
        let _ = CoInitializeEx(std::ptr::null_mut(), 0x2 | 0x4);
        let mut bi = BROWSEINFOW {
            hwndOwner: owner,
            pidlRoot: std::ptr::null_mut(),
            pszDisplayName: std::ptr::null_mut(),
            lpszTitle: title_w.as_ptr(),
            ulFlags: FLAGS,
            lpfn: start_w.as_ref().map(|_| browse_callback as _),
            lParam: start_w.as_ref().map_or(0, |v| v.as_ptr() as isize),
            iImage: 0,
        };
        let pidl = SHBrowseForFolderW(&mut bi);
        if pidl.is_null() {
            return None;
        }
        let mut buf = [0u16; 260];
        let ok = SHGetPathFromIDListW(pidl, buf.as_mut_ptr());
        CoTaskMemFree(pidl);
        if ok == 0 {
            return None;
        }
        let n = buf.iter().position(|c| *c == 0).unwrap_or(buf.len());
        Some(String::from_utf16_lossy(&buf[..n]))
    }
}

/// Preselects the folder the setting already names.
///
/// `BFFM_INITIALIZED` is 1 and `BFFM_SETSELECTIONW` is 0x467.
unsafe extern "system" fn browse_callback(hwnd: HWND, msg: u32, _l: isize, data: isize) -> i32 {
    if msg == 1 && data != 0 {
        SendMessageW(hwnd, 0x467, 1, data);
    }
    0
}

/// Screen coordinates of the primary monitor's **work area** — the desktop
/// minus the taskbar.
pub fn work_area() -> Option<(i32, i32, i32, i32)> {
    let mut r = RECT::default();
    // SPI_GETWORKAREA = 0x0030. SAFETY: writes into a local RECT.
    let ok = unsafe { SystemParametersInfoW(0x0030, 0, &mut r as *mut RECT as *mut c_void, 0) };
    (ok != 0).then_some((r.left, r.top, r.right, r.bottom))
}

/// Tell Windows this process draws its own pixels at the monitor's real
/// resolution.
///
/// # Why a window can open in the wrong place without this
///
/// A process that says nothing is **DPI-virtualised**: on a display at 125% it
/// renders into a 96-DPI surface and the system stretches the result. Text goes
/// soft, and every coordinate the app computes is in a made-up space — a screen
/// reported as 1536x864 that is really 1920x1080. So a window asked for at
/// (120, 80) does not land at (120, 80), and a size chosen to fit the desktop
/// does not fit it.
///
/// Per-monitor v2 is the mode that also keeps this right when the window is
/// dragged between monitors of different scaling, which is the case on the
/// machine this was found on: 1920x1080 beside a scaled 1536x864.
///
/// Called before any window exists, because awareness cannot be changed
/// afterwards. Best effort: on Windows 8.1 and older the newer entry points are
/// absent, and `SetProcessDPIAware` is the whole of what is available.
pub fn become_dpi_aware() {
    // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4.
    const PER_MONITOR_V2: isize = -4;
    // SAFETY: both are pure process-wide settings taking no pointers we own.
    unsafe {
        let user32 = LoadLibraryW(wide_z("user32.dll").as_ptr());
        if !user32.is_null() {
            // A C string literal, so the NUL is in the constant and cannot be
            // forgotten -- `GetProcAddress` reads until one.
            let name = c"SetProcessDpiAwarenessContext";
            let f = GetProcAddress(user32, name.as_ptr());
            if !f.is_null() {
                let f: extern "system" fn(isize) -> BOOL = std::mem::transmute(f);
                if f(PER_MONITOR_V2) != 0 {
                    return;
                }
            }
        }
        SetProcessDPIAware();
    }
}

fn wide_z(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// The working set of another process, in bytes, or `None` if it cannot be
/// asked.
///
/// **This is how a model's loading progress is measured.** `chaos-serve` reads
/// its always-read weights into memory before it answers anything, so its
/// working set against the catalogue's resident figure is a percentage that
/// needs no cooperation from the process being watched — the same reasoning
/// that made bytes-on-disk the download's progress.
pub fn working_set(pid: u32) -> Option<u64> {
    // SAFETY: a null handle is checked; every pointer is to a local.
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid);
        if h.is_null() {
            return None;
        }
        let mut c = PROCESS_MEMORY_COUNTERS {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..Default::default()
        };
        let ok = K32GetProcessMemoryInfo(h, &mut c, c.cb);
        CloseHandle(h);
        (ok != 0).then_some(c.WorkingSetSize as u64)
    }
}

/// `SHCNE_ASSOCCHANGED`: "what a file looks like has changed."
///
/// Explorer keeps a database of the icons it has already seen, keyed by file
/// path, and it does **not** re-read an executable that has been overwritten.
/// So a new version with a new icon shows the old one until the cache is
/// rebuilt, which is why "the app icon is still the old one for me" is a real
/// report about a correct file. This is the documented way to say otherwise.
pub const SHCNE_ASSOCCHANGED: i32 = 0x0800_0000;
pub const SHCNF_IDLIST: u32 = 0x0000;

#[link(name = "shell32")]
extern "system" {
    pub fn SHChangeNotify(
        wEventId: i32,
        uFlags: u32,
        dwItem1: *const c_void,
        dwItem2: *const c_void,
    );
}

#[link(name = "kernel32")]
extern "system" {
    pub fn GlobalAlloc(uFlags: u32, dwBytes: usize) -> *mut c_void;
    pub fn GlobalLock(hMem: *mut c_void) -> *mut c_void;
    pub fn GlobalUnlock(hMem: *mut c_void) -> BOOL;
}

/// `EM_SETMARGINS`: an `EDIT` puts its text flush against the border otherwise,
/// which on a design built out of whitespace is the one control that has none.
pub const EM_SETMARGINS: u32 = 0x00D3;
pub const EC_LEFTMARGIN: WPARAM = 0x0001;
pub const EC_RIGHTMARGIN: WPARAM = 0x0002;

#[link(name = "shell32")]
extern "system" {
    fn ShellExecuteW(
        hwnd: HWND,
        lpOperation: *const u16,
        lpFile: *const u16,
        lpParameters: *const u16,
        lpDirectory: *const u16,
        nShowCmd: i32,
    ) -> *mut c_void;
}

/// Hand something to the shell to open -- a folder in Explorer, a URL in the
/// browser. Wrapped so no caller has to build five wide strings for the two
/// arguments that ever vary.
pub fn shell_open(target: &str) {
    unsafe {
        let op = wide("open");
        let f = wide(target);
        ShellExecuteW(
            std::ptr::null_mut(),
            op.as_ptr(),
            f.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            SW_SHOW,
        );
    }
}

/// Put text on the clipboard.
///
/// The endpoint is the single most copied string in this app -- it is what you
/// paste into a coding agent -- and retyping `http://127.0.0.1:8231/v1` from a
/// screenshot is exactly the kind of small friction that makes a window feel
/// like a demo.
///
/// Returns whether it worked, because a clipboard held open by another process
/// is a real and silent failure.
///
/// # Safety
/// `hwnd` is handed to `OpenClipboard`, which will use it as a window handle.
/// A safe function taking a raw pointer and passing it to Windows is a safe
/// function that dereferences whatever it is given -- the same reason
/// `hkcu_read_string` refuses to take an `HKEY`.
pub unsafe fn set_clipboard_text(hwnd: HWND, text: &str) -> bool {
    let utf16 = wide(text);
    unsafe {
        if OpenClipboard(hwnd) == 0 {
            return false;
        }
        EmptyClipboard();
        let bytes = utf16.len() * 2;
        let h = GlobalAlloc(GMEM_MOVEABLE, bytes);
        if h.is_null() {
            CloseClipboard();
            return false;
        }
        let p = GlobalLock(h);
        if p.is_null() {
            CloseClipboard();
            return false;
        }
        std::ptr::copy_nonoverlapping(utf16.as_ptr(), p as *mut u16, utf16.len());
        GlobalUnlock(h);
        // On success the clipboard owns the block; freeing it here would be a
        // double free the next paste discovers.
        let ok = !SetClipboardData(CF_UNICODETEXT, h).is_null();
        CloseClipboard();
        ok
    }
}

// -- dark mode beyond the client area ----------------------------------------
//
// `DwmSetWindowAttribute` darkens the title bar and nothing else. A dark Chaos
// came up with light scrollbars, measured at `#F0F0F0` against a `#0D0D0E`
// page, because a scrollbar is drawn by the system from the control's theme
// class rather than from anything the parent says.
//
// The menu bar has no such fix and is still light in dark mode -- see the note
// in `main.rs`'s `sync_titlebar`, which records what was tried and measured.

#[link(name = "uxtheme")]
extern "system" {
    /// Documented. Naming a control's theme class is how a scrollbar is told
    /// to draw itself dark.
    pub fn SetWindowTheme(hwnd: HWND, sub_app: *const u16, sub_id: *const u16) -> i32;
}

/// Put one control's scrollbars into the matching theme.
///
/// # Safety
/// `hwnd` is passed to `SetWindowTheme`, which treats it as a live window.
pub unsafe fn set_control_theme(hwnd: HWND, dark: bool) {
    if hwnd.is_null() {
        return;
    }
    let name = wide(if dark {
        "DarkMode_Explorer"
    } else {
        "Explorer"
    });
    unsafe {
        SetWindowTheme(hwnd, name.as_ptr(), std::ptr::null());
    }
}

// -- picking a face that actually exists -------------------------------------

#[link(name = "gdi32")]
extern "system" {
    /// The face GDI *actually* selected, which is not always the one asked for.
    pub fn GetTextFaceW(hdc: HDC, c: i32, name: *mut u16) -> i32;
}

#[link(name = "user32")]
extern "system" {
    /// A device context to measure against before there is anything to paint.
    /// Must be released, not deleted -- `DeleteDC` is for the ones *we* create.
    pub fn GetDC(hWnd: HWND) -> HDC;
    pub fn ReleaseDC(hWnd: HWND, hDC: HDC) -> i32;
}

/// The first of `wanted` that is installed, or `None`.
///
/// **`CreateFontW` never fails.** Ask for a face that is not installed and GDI
/// silently substitutes something else, so a display serif chosen for a wordmark
/// can quietly become the default UI font and nothing says so. The only way to
/// find out is to select the font into a DC and ask what came back.
///
/// # Safety
/// `hdc` must be a live device context; the font is selected into it and
/// restored before returning.
pub unsafe fn first_available_face(hdc: HDC, wanted: &[&str]) -> Option<String> {
    for name in wanted {
        let f = CreateFontW(
            -20,
            0,
            0,
            0,
            400,
            0,
            0,
            0,
            1,
            0,
            0,
            5,
            0,
            wide(name).as_ptr(),
        );
        if f.is_null() {
            continue;
        }
        let old = SelectObject(hdc, f as HGDIOBJ);
        let mut buf = [0u16; 64];
        let n = GetTextFaceW(hdc, buf.len() as i32, buf.as_mut_ptr());
        SelectObject(hdc, old);
        DeleteObject(f as HGDIOBJ);
        if n > 0 {
            let got = String::from_utf16_lossy(&buf[..(n as usize).saturating_sub(1)]);
            if got.eq_ignore_ascii_case(name) {
                return Some(got);
            }
        }
    }
    None
}

// -- combo boxes -------------------------------------------------------------
//
// A settings page made of empty text boxes asks a question most people cannot
// answer. These are the dropdowns that replace them, owner-drawn for the same
// reason the buttons are: a themed combo ignores `WM_CTLCOLOR*`.

/// A list that cannot be typed into: the choices are the choices.
pub const CBS_DROPDOWNLIST: u32 = 0x0003;
pub const CBS_OWNERDRAWFIXED: u32 = 0x0010;
pub const CBS_HASSTRINGS: u32 = 0x0200;

pub const CB_ADDSTRING: u32 = 0x0143;
/// Widen the *open* list past the closed control.
///
/// Without it the drop-down is exactly as wide as the box, so an option whose
/// label does not fit is unreadable in the one place it has to be read -- which
/// is what "Processor (the GPU is not used her..." was.
pub const CB_SETDROPPEDWIDTH: u32 = 0x0160;
pub const CB_RESETCONTENT: u32 = 0x014B;
pub const CB_SETCURSEL: u32 = 0x014E;
pub const CB_GETCURSEL: u32 = 0x0147;
pub const CB_SETITEMHEIGHT: u32 = 0x0153;
pub const CBN_SELCHANGE: u16 = 1;

pub const ODT_COMBOBOX: u32 = 3;
pub const ODS_COMBOBOXEDIT: u32 = 0x1000;

/// Windows asks for a row height before it draws one.
pub const WM_MEASUREITEM: u32 = 0x002C;

#[repr(C)]
pub struct MEASUREITEMSTRUCT {
    pub CtlType: u32,
    pub CtlID: u32,
    pub itemID: u32,
    pub itemWidth: u32,
    pub itemHeight: u32,
    pub itemData: usize,
}

// -- randomness --------------------------------------------------------------

#[link(name = "bcrypt")]
extern "system" {
    fn BCryptGenRandom(alg: *mut c_void, buf: *mut u8, len: u32, flags: u32) -> i32;
}

/// `BCRYPT_USE_SYSTEM_PREFERRED_RNG`: no algorithm handle to open or close.
const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;

/// `n` random bytes as lowercase hex, from the system generator.
///
/// **Not `SystemTime` mixed with a process id.** An API key derived from the
/// clock is guessable by anyone who knows roughly when it was made, and the
/// whole point of the key is that it cannot be guessed. Windows ships a CSPRNG;
/// this is the two-line way to ask it.
///
/// Returns `None` rather than falling back to something weaker: a key that
/// silently is not random is worse than no key, because it is trusted.
pub fn random_hex(n: usize) -> Option<String> {
    let mut buf = vec![0u8; n];
    let ok = unsafe {
        BCryptGenRandom(
            std::ptr::null_mut(),
            buf.as_mut_ptr(),
            n as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    // STATUS_SUCCESS is 0; anything else is a failure worth surfacing.
    (ok == 0).then(|| buf.iter().map(|b| format!("{b:02x}")).collect())
}
