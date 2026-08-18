//! Chaos setup.
//!
//! One .exe that carries every binary inside it. Double-click, pick a folder,
//! press INSTALL. No archive to unpack, no PowerShell to run, no toolchain, no
//! network -- the payload is linked in, so this works on a machine that has
//! never had internet.
//!
//! Built without NSIS, WiX, Inno or MSI tooling, because a Windows install is a
//! window, a file copy, a PATH entry, a shortcut and one registry key, and all
//! of those are in libraries Windows already ships. Taking installer tooling
//! would mean it had to be present on the build machine before a release could
//! be cut, and this project's rule is that it has no dependencies.
//!
//! Same two colours as the app.

#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(not(windows))]
fn main() {
    eprintln!("chaos-setup installs Chaos on Windows; elsewhere copy the binaries onto PATH.");
    std::process::exit(1);
}

#[cfg(windows)]
mod payload {
    include!(concat!(env!("OUT_DIR"), "/payload.rs"));
}

#[cfg(windows)]
fn main() {
    // Silent mode, the convention every Windows installer follows -- and the
    // only way this can be exercised by CI, which has no one to press a button.
    // `/S` installs, `/S /uninstall` removes, both without a window.
    //
    // **A caller must WAIT for this process.** The binary is built for the
    // window subsystem, so a shell that starts it gets control back immediately
    // and sees no exit code at all -- in PowerShell `$LASTEXITCODE` is left
    // empty, and the natural `if ($LASTEXITCODE -ne 0) { throw }` fires because
    // `$null -ne 0`. That failed a release once while the installer was working
    // perfectly; it simply had not been given time to start. Use
    // `Start-Process -Wait -PassThru` and read `.ExitCode`, or `cmd /c start
    // /wait`. NSIS and every other GUI installer behave the same way.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let silent = args
        .iter()
        .any(|a| a.eq_ignore_ascii_case("/S") || a == "--silent");
    if silent {
        let uninstall = args
            .iter()
            .any(|a| a.eq_ignore_ascii_case("/uninstall") || a == "--uninstall");
        let prefix = args
            .iter()
            .position(|a| a == "--prefix")
            .and_then(|i| args.get(i + 1))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(chaos_setup::default_prefix);
        let (ok, msg) = if uninstall {
            setup::uninstall_to(&prefix)
        } else {
            setup::install_to(&prefix)
        };
        // A silent installer must still say what happened somewhere, and a
        // windows-subsystem binary has no console -- so the message goes to the
        // exit code and to a log.
        //
        // **Not into the prefix when uninstalling.** Writing it there recreates
        // the directory that was just removed, so a "clean" uninstall left an
        // empty `Chaos` folder holding one log file saying it had uninstalled.
        let log = if uninstall {
            std::env::temp_dir().join("chaos-uninstall.log")
        } else {
            prefix.join("setup.log")
        };
        let _ = std::fs::write(log, &msg);
        std::process::exit(if ok { 0 } else { 1 });
    }
    setup::run();
}

#[cfg(windows)]
mod setup {
    use crate::payload;
    use chaos_app::win32::*;
    use chaos_setup::*;
    use std::cell::RefCell;
    use std::path::PathBuf;
    use std::process::Command;

    const ID_PREFIX: i32 = 201;
    const ID_INSTALL: i32 = 202;
    const ID_UNINSTALL: i32 = 203;

    const W: i32 = 720;
    const H: i32 = 460;
    const PAD: i32 = 22;
    const LOGO_PX: i32 = 72;

    struct S {
        main: HWND,
        prefix: HWND,
        font: HFONT,
        mono: HFONT,
        black: HBRUSH,
        status: String,
        done: bool,
    }

    thread_local! {
        static S: RefCell<Option<S>> = const { RefCell::new(None) };
    }

    fn set_status(msg: &str) {
        S.with(|s| {
            if let Some(s) = s.borrow_mut().as_mut() {
                s.status = msg.to_string();
                unsafe {
                    InvalidateRect(s.main, std::ptr::null(), 1);
                    UpdateWindow(s.main);
                }
            }
        });
    }

    pub fn run() {
        unsafe {
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("ChaosSetupWindow");
            let black = CreateSolidBrush(BLACK);
            let wc = WNDCLASSW {
                style: 0,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinst,
                hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW as *const u16),
                hbrBackground: black,
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
            };
            if RegisterClassW(&wc) == 0 {
                return;
            }
            let title = wide("Chaos Setup");
            // No WS_THICKFRAME or maximise: the layout is fixed, and a resizable
            // window that does not reflow is worse than one that cannot resize.
            let style = 0x00CA_0000u32; // WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                style,
                200,
                160,
                W,
                H,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinst,
                std::ptr::null_mut(),
            );
            if hwnd.is_null() {
                return;
            }
            let on: i32 = 1;
            DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                &on as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u32,
            );

            let font = CreateFontW(
                -15,
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
                wide("Segoe UI").as_ptr(),
            );
            let mono = CreateFontW(
                -13,
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
                wide("Consolas").as_ptr(),
            );

            let prefix_text = wide(&default_prefix().to_string_lossy());
            let prefix = CreateWindowExW(
                0,
                wide("EDIT").as_ptr(),
                prefix_text.as_ptr(),
                WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
                PAD,
                PAD * 2 + LOGO_PX + 44,
                W - PAD * 2 - 16,
                28,
                hwnd,
                ID_PREFIX as HMENU,
                hinst,
                std::ptr::null_mut(),
            );
            let mk = |label: &str, id: i32, x: i32| {
                CreateWindowExW(
                    0,
                    wide("BUTTON").as_ptr(),
                    wide(label).as_ptr(),
                    WS_CHILD | WS_VISIBLE | BS_OWNERDRAW | WS_TABSTOP,
                    x,
                    H - 118,
                    170,
                    36,
                    hwnd,
                    id as HMENU,
                    hinst,
                    std::ptr::null_mut(),
                )
            };
            // Held only long enough to set their font: after that they are
            // reached by id from WM_COMMAND, which is where their work happens.
            let install = mk("INSTALL", ID_INSTALL, PAD);
            let uninstall = mk("UNINSTALL", ID_UNINSTALL, PAD + 186);

            for h in [prefix, install, uninstall] {
                SendMessageW(h, WM_SETFONT, font as WPARAM, 1);
            }
            SendMessageW(prefix, WM_SETFONT, mono as WPARAM, 1);

            S.with(|s| {
                *s.borrow_mut() = Some(S {
                    main: hwnd,
                    prefix,
                    font,
                    mono,
                    black,
                    status: if payload::FILES.is_empty() {
                        "this installer was built with no payload".into()
                    } else {
                        format!("{} files ready to install", payload::FILES.len())
                    },
                    done: false,
                })
            });

            ShowWindow(hwnd, SW_SHOW);
            UpdateWindow(hwnd);
            let mut msg = MSG::default();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    fn prefix_value() -> PathBuf {
        S.with(|s| {
            let b = s.borrow();
            let Some(s) = b.as_ref() else {
                return default_prefix();
            };
            unsafe {
                let n = GetWindowTextLengthW(s.prefix);
                if n <= 0 {
                    return default_prefix();
                }
                let mut buf = vec![0u16; n as usize + 1];
                let got = GetWindowTextW(s.prefix, buf.as_mut_ptr(), n + 1);
                PathBuf::from(String::from_utf16_lossy(&buf[..got.max(0) as usize]))
            }
        })
    }

    fn do_install() {
        let prefix = prefix_value();
        let (_, msg) = install_to(&prefix);
        S.with(|s| {
            if let Some(s) = s.borrow_mut().as_mut() {
                s.done = true;
            }
        });
        set_status(&msg);
    }

    fn do_uninstall() {
        let prefix = prefix_value();
        let (_, msg) = uninstall_to(&prefix);
        set_status(&msg);
    }

    /// Install into `prefix`. Returns success and a message for either mode.
    pub fn install_to(prefix: &std::path::Path) -> (bool, String) {
        if payload::FILES.is_empty() {
            return (
                false,
                "nothing to install: this build carries no payload".into(),
            );
        }
        let prefix = prefix.to_path_buf();
        let bin = bin_dir(&prefix);
        if let Err(e) = std::fs::create_dir_all(&bin) {
            return (false, format!("cannot create {}: {e}", bin.display()));
        }
        let models = default_models_dir();
        let _ = std::fs::create_dir_all(&models);

        // Remove what a previous version installed and this one does not, so an
        // upgrade cannot leave a stale binary on PATH.
        let incoming: Vec<String> = payload::FILES.iter().map(|f| f.name.to_string()).collect();
        if let Ok(prev) = std::fs::read_to_string(manifest_path(&prefix)) {
            let prev: Vec<String> = prev
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            for old in stale(&prev, &incoming) {
                let _ = std::fs::remove_file(bin.join(old));
            }
        }

        for f in payload::FILES.iter() {
            let dest = bin.join(f.name);
            if let Err(e) = std::fs::write(&dest, f.bytes) {
                // A running binary is locked, and this is the one failure a
                // user hits twice: install, run, install again.
                return (
                    false,
                    format!(
                        "cannot write {} ({e}). Close Chaos and run this again.",
                        f.name
                    ),
                );
            }
        }
        let _ = std::fs::write(manifest_path(&prefix), incoming.join("\n"));

        add_to_path(&bin.to_string_lossy());
        make_shortcut(&bin);
        register_uninstall(&prefix);

        (
            true,
            format!(
                "installed {} files. Open a new terminal, or find Chaos in the Start Menu. Models: {}",
                payload::FILES.len(),
                models.display()
            ),
        )
    }

    /// Is this executable inside the directory being uninstalled?
    fn running_inside(prefix: &std::path::Path) -> bool {
        let Ok(me) = std::env::current_exe() else {
            return false;
        };
        // Canonicalise both: `%LOCALAPPDATA%\Chaos` and the resolved path can
        // differ by case or by a junction, and a false negative here brings the
        // whole bug back.
        let me = me.canonicalize().unwrap_or(me);
        let prefix = prefix
            .canonicalize()
            .unwrap_or_else(|_| prefix.to_path_buf());
        me.starts_with(&prefix)
    }

    /// Copy this executable to the temp directory and let that copy uninstall.
    ///
    /// **Spawned detached, and deliberately not waited for.** Waiting is what
    /// made the first attempt at this fail: `cmd.status()` keeps the parent --
    /// which lives inside the directory being deleted -- alive for the whole of
    /// the child's run, so Windows still held the file open and the child could
    /// not remove it either. The parent has to be gone. So it starts the child
    /// and returns immediately, and the child retries until the lock clears.
    ///
    /// This is how NSIS uninstallers behave too, for exactly this reason.
    fn relaunch_from_temp(prefix: &std::path::Path) -> Result<String, String> {
        let me = std::env::current_exe().map_err(|e| e.to_string())?;
        // A fixed name rather than a random one: this workspace has no random
        // number generator, a stale copy is simply overwritten, and it sits in
        // a directory Windows cleans up.
        let tmp = std::env::temp_dir().join("chaos-uninstall.exe");
        std::fs::copy(&me, &tmp).map_err(|e| format!("cannot stage the uninstaller: {e}"))?;

        let mut cmd = Command::new(&tmp);
        cmd.arg("/S").arg("--uninstall").arg("--prefix").arg(prefix);
        {
            use std::os::windows::process::CommandExt;
            // DETACHED_PROCESS too: the child must outlive this one.
            cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
        }
        cmd.spawn()
            .map_err(|e| format!("cannot start the staged uninstaller: {e}"))?;
        Ok("uninstalling -- the files are removed a moment after this closes.".into())
    }

    /// Delete a directory, waiting out a lock rather than giving up on it.
    ///
    /// The process that asked for the uninstall may still be exiting, and
    /// Windows keeps a running executable's file open until it has. A single
    /// attempt fails in that window; a few seconds of retries does not.
    fn remove_dir_all_retrying(dir: &std::path::Path) -> bool {
        for _ in 0..40 {
            if !dir.exists() {
                return true;
            }
            if std::fs::remove_dir_all(dir).is_ok() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        !dir.exists()
    }

    /// Remove an install from `prefix`.
    pub fn uninstall_to(prefix: &std::path::Path) -> (bool, String) {
        let prefix = prefix.to_path_buf();
        let bin = bin_dir(&prefix);

        // **A running executable cannot delete the directory it lives in.**
        // The installer copies itself into `bin` so Add/Remove Programs has
        // something to launch -- which means the normal uninstall runs from
        // inside the very folder it is trying to remove, Windows holds the file
        // open, `remove_dir_all` fails, and the old code reported success
        // anyway. Uninstalling left every binary on disk and said "uninstalled".
        //
        // So: if this copy is inside the prefix, re-run from a copy in the temp
        // directory and let that one do the work. The relaunched process is
        // holding a file nothing is about to delete.
        if running_inside(&prefix) {
            match relaunch_from_temp(&prefix) {
                Ok(msg) => return (true, msg),
                // Falling through is still better than stopping: everything
                // except this one executable can be removed from here.
                Err(e) => {
                    let _ =
                        std::fs::write(prefix.join("setup.log"), format!("relaunch failed: {e}"));
                }
            }
        }

        remove_dir_all_retrying(&bin);
        let _ = std::fs::remove_file(manifest_path(&prefix));
        let _ = std::fs::remove_file(prefix.join("setup.log"));
        // Only if empty: the prefix may be a directory the user chose and
        // already had things of their own in.
        let _ = std::fs::remove_dir(&prefix);
        remove_from_path(&bin.to_string_lossy());
        let _ = std::fs::remove_file(start_menu_lnk());
        unregister_uninstall();

        // Report what is actually true. `remove_dir_all` is best-effort here --
        // a binary still running keeps its own file alive -- so check rather
        // than assume, because "uninstalled" over a full directory is the lie
        // this function used to tell.
        if bin.exists() {
            let left = std::fs::read_dir(&bin).map(|d| d.count()).unwrap_or(0);
            return (
                false,
                format!(
                    "removed the PATH entry, shortcut and registry entry, but {left} file(s)                      are still in {} because something has them open. Close anything                      running from there and uninstall again.",
                    bin.display()
                ),
            );
        }
        (
            true,
            "uninstalled. Your models were left where they are.".into(),
        )
    }

    // -- registry ------------------------------------------------------------

    fn add_to_path(dir: &str) {
        let cur = read_user_path();
        let next = path_with(&cur, dir);
        if next != cur {
            write_user_path(&next);
        }
    }

    fn remove_from_path(dir: &str) {
        let cur = read_user_path();
        let next = path_without(&cur, dir);
        if next != cur {
            write_user_path(&next);
        }
    }

    /// Read the *user* PATH from the registry, never the process environment.
    ///
    /// The process variable is the merge of machine and user PATH plus whatever
    /// the parent shell exported, so testing against it would refuse to add an
    /// entry that is not actually persisted -- and the install would work until
    /// the next reboot.
    fn read_user_path() -> String {
        hkcu_read_string("Environment", "Path").unwrap_or_default()
    }

    fn write_user_path(v: &str) {
        hkcu_write_string("Environment", "Path", v);
        // Tell running shells. Without this nothing sees the new PATH until a
        // logout, and the first thing a user does is open a terminal.
        unsafe {
            let param = wide("Environment");
            let mut out: usize = 0;
            SendMessageTimeoutW(
                HWND_BROADCAST,
                WM_SETTINGCHANGE,
                0,
                param.as_ptr() as LPARAM,
                SMTO_ABORTIFHUNG,
                2000,
                &mut out,
            );
        }
    }

    /// The Add/Remove Programs entry, so Chaos appears in Settings like any
    /// other installed application rather than only in a folder.
    fn register_uninstall(prefix: &std::path::Path) {
        let key = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Chaos";
        let exe = prefix.join("bin").join("chaos-setup.exe");
        // The installer copies itself in so uninstall has something to run.
        if let Ok(me) = std::env::current_exe() {
            let _ = std::fs::copy(me, &exe);
        }
        hkcu_write_string(key, "DisplayName", "Chaos");
        hkcu_write_string(key, "DisplayVersion", env!("CARGO_PKG_VERSION"));
        hkcu_write_string(key, "Publisher", "aturzone");
        hkcu_write_string(key, "InstallLocation", &prefix.to_string_lossy());
        hkcu_write_string(key, "UninstallString", &format!("\"{}\"", exe.display()));
        hkcu_write_string(key, "URLInfoAbout", "https://github.com/aturzone/Chaos");
    }

    fn unregister_uninstall() {
        hkcu_delete_key(r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Chaos");
    }

    fn start_menu_lnk() -> PathBuf {
        let appdata = std::env::var("APPDATA").unwrap_or_default();
        PathBuf::from(appdata)
            .join(r"Microsoft\Windows\Start Menu\Programs")
            .join("Chaos.lnk")
    }

    /// Create the Start Menu shortcut.
    ///
    /// A `.lnk` is a COM object (`IShellLink` + `IPersistFile`), and driving COM
    /// by hand through raw vtables for one shortcut is a great deal of unsafe
    /// code for a file nobody will inspect. `WScript.Shell` does it in three
    /// lines and ships with every Windows install, so this shells out and
    /// treats failure as cosmetic -- the app is on PATH either way.
    fn make_shortcut(bin: &std::path::Path) {
        let lnk = start_menu_lnk();
        let target = bin.join("chaos-app.exe");
        if !target.exists() {
            return;
        }
        let script = format!(
            "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{}');\
             $s.TargetPath='{}';$s.WorkingDirectory='{}';$s.Description='Chaos';$s.Save()",
            lnk.display(),
            target.display(),
            bin.display()
        );
        let mut cmd = std::process::Command::new("powershell");
        cmd.args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &script,
        ]);
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let _ = cmd.status();
    }

    // -- painting ------------------------------------------------------------

    unsafe fn paint(hwnd: HWND) {
        let mut ps: PAINTSTRUCT = std::mem::zeroed();
        let hdc = BeginPaint(hwnd, &mut ps);
        let mut r = RECT::default();
        GetClientRect(hwnd, &mut r);
        S.with(|s| {
            let b = s.borrow();
            let Some(s) = b.as_ref() else { return };
            FillRect(hdc, &r, s.black);
            SetBkMode(hdc, TRANSPARENT);
            SetTextColor(hdc, WHITE);

            let (lw, lh) = chaos_app::art::logo_size();
            let mono_bits = chaos_app::art::logo_mono();
            let mut px = vec![0u8; lw * lh * 4];
            for y in 0..lh {
                for x in 0..lw {
                    let v = if mono_bits[(lh - 1 - y) * lw + x] {
                        255u8
                    } else {
                        0u8
                    };
                    let i = (y * lw + x) * 4;
                    px[i] = v;
                    px[i + 1] = v;
                    px[i + 2] = v;
                }
            }
            let bmi = BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: lw as i32,
                biHeight: lh as i32,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB,
                biSizeImage: 0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: 0,
                biClrImportant: 0,
            };
            StretchDIBits(
                hdc,
                PAD,
                PAD,
                LOGO_PX,
                LOGO_PX,
                0,
                0,
                lw as i32,
                lh as i32,
                px.as_ptr() as *const _,
                &bmi,
                DIB_RGB_COLORS,
                SRCCOPY,
            );

            SelectObject(hdc, s.font as HGDIOBJ);
            let t = |x: i32, y: i32, txt: &str| {
                let w = wide(txt);
                TextOutW(hdc, x, y, w.as_ptr(), txt.encode_utf16().count() as i32);
            };
            t(PAD + LOGO_PX + 18, PAD + 6, "C H A O S");
            t(
                PAD + LOGO_PX + 18,
                PAD + 30,
                "Run models that do not fit in your RAM.",
            );
            t(PAD, PAD * 2 + LOGO_PX + 18, "Install to:");

            SelectObject(hdc, s.mono as HGDIOBJ);
            t(PAD, H - 168, &s.status);
            if s.done {
                t(
                    PAD,
                    H - 148,
                    "chaos-app   the window    chaos-run   the command line",
                );
            } else {
                t(
                    PAD,
                    H - 148,
                    "Per-user install. No administrator rights needed.",
                );
            }
        });
        EndPaint(hwnd, &ps);
    }

    unsafe fn draw_item(di: &DRAWITEMSTRUCT) {
        let selected = di.itemState & ODS_SELECTED != 0;
        let disabled = di.itemState & ODS_DISABLED != 0;
        let (bg, fg) = if selected {
            (WHITE, BLACK)
        } else {
            (BLACK, WHITE)
        };
        let brush = CreateSolidBrush(bg);
        FillRect(di.hDC, &di.rcItem, brush);
        DeleteObject(brush as HGDIOBJ);
        if !disabled {
            let pen = CreatePen(0, 1, fg);
            let op = SelectObject(di.hDC, pen);
            let hollow = CreateSolidBrush(bg);
            let ob = SelectObject(di.hDC, hollow as HGDIOBJ);
            Rectangle(
                di.hDC,
                di.rcItem.left,
                di.rcItem.top,
                di.rcItem.right,
                di.rcItem.bottom,
            );
            SelectObject(di.hDC, ob);
            SelectObject(di.hDC, op);
            DeleteObject(hollow as HGDIOBJ);
            DeleteObject(pen as HGDIOBJ);
        }
        let n = GetWindowTextLengthW(di.hwndItem);
        if n <= 0 {
            return;
        }
        let mut buf = vec![0u16; n as usize + 1];
        let got = GetWindowTextW(di.hwndItem, buf.as_mut_ptr(), n + 1);
        let text = String::from_utf16_lossy(&buf[..got.max(0) as usize]);
        SetBkMode(di.hDC, TRANSPARENT);
        SetTextColor(di.hDC, fg);
        let w: Vec<u16> = text.encode_utf16().collect();
        let cw = (di.rcItem.right - di.rcItem.left) / 2;
        let approx = (text.len() as i32 * 7) / 2;
        TextOutW(
            di.hDC,
            di.rcItem.left + (cw - approx).max(8),
            di.rcItem.top + (di.rcItem.bottom - di.rcItem.top) / 2 - 10,
            w.as_ptr(),
            w.len() as i32,
        );
    }

    unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        match msg {
            WM_PAINT => {
                paint(hwnd);
                0
            }
            WM_DRAWITEM => {
                draw_item(&*(lp as *const DRAWITEMSTRUCT));
                1
            }
            WM_CTLCOLOREDIT | WM_CTLCOLORSTATIC | WM_CTLCOLORBTN => {
                SetTextColor(wp as HDC, WHITE);
                SetBkColor(wp as HDC, BLACK);
                S.with(|s| s.borrow().as_ref().map(|s| s.black as LRESULT).unwrap_or(0))
            }
            WM_COMMAND => {
                let id = (wp & 0xFFFF) as i32;
                if ((wp >> 16) & 0xFFFF) as u16 == BN_CLICKED {
                    match id {
                        ID_INSTALL => do_install(),
                        ID_UNINSTALL => do_uninstall(),
                        _ => {}
                    }
                }
                0
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                0
            }
            _ => DefWindowProcW(hwnd, msg, wp, lp),
        }
    }
}
