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

/// Make a crash say something.
///
/// **The installer had none of this, and it cost an evening.** The release
/// profile sets `panic = "abort"`, so a panic is not an unwind that can be
/// caught -- the process simply vanishes, and with no console attached there is
/// no message, no log and nothing to report. That is exactly how the first run
/// of the stepped install presented: the window disappeared mid-install and the
/// only evidence was a half-written directory. The app has had this since the
/// same failure happened there; the installer is the program that can least
/// afford to fail silently, on a machine that is not yours.
#[cfg(windows)]
fn install_panic_hook() {
    use chaos_app::win32::{wide, MessageBoxW, MB_ICONERROR, MB_OK};
    std::panic::set_hook(Box::new(|info| {
        let where_ = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "unknown".into());
        let what = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panic".into());
        let text = format!(
            "Chaos Setup failed.

{what}
at {where_}
"
        );
        let path = std::env::temp_dir().join("chaos-setup-crash.log");
        let _ = std::fs::write(&path, &text);
        unsafe {
            let msg = format!(
                "{text}
Written to:
{}",
                path.display()
            );
            MessageBoxW(
                std::ptr::null_mut(),
                wide(&msg).as_ptr(),
                wide("Chaos Setup").as_ptr(),
                MB_OK | MB_ICONERROR,
            );
        }
    }));
}

#[cfg(windows)]
fn main() {
    install_panic_hook();
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
    use chaos_app::theme::{self, Theme};
    use chaos_app::win32::*;
    use chaos_setup::*;
    use chaos_setup::{human_millis, Progress, Step, StepState};
    use std::cell::RefCell;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    const ID_PREFIX: i32 = 201;
    const ID_INSTALL: i32 = 202;
    const ID_UNINSTALL: i32 = 203;

    // Hermes' installer is a wide, calm field with the content in the middle
    // third. 900x680 gives the step list room without a scrollbar.
    const W: i32 = 900;
    const H: i32 = 680;
    const PAD: i32 = 40;

    /// Which screen the installer is on.
    #[derive(Clone, Copy, PartialEq)]
    enum Screen {
        /// A wordmark, one sentence, and one action. Nothing else.
        Welcome,
        /// The named step list, with a bar and per-step timings.
        Working,
        /// The report, good or bad.
        Report,
    }

    struct S {
        main: HWND,
        prefix: HWND,
        screen: Screen,
        theme: Theme,
        display: HFONT,
        font: HFONT,
        bold: HFONT,
        small: HFONT,
        mono: HFONT,
        ground: HBRUSH,
        status: String,
        done: bool,
    }

    thread_local! {
        static S: RefCell<Option<S>> = const { RefCell::new(None) };
    }

    /// The install's progress, written by the worker and read by the window.
    ///
    /// The install used to run on the UI thread, so the window was frozen for
    /// its whole duration and said nothing about what it was doing. It runs on
    /// a worker now and reports here.
    fn progress() -> &'static std::sync::Mutex<Progress> {
        static P: std::sync::OnceLock<std::sync::Mutex<Progress>> = std::sync::OnceLock::new();
        P.get_or_init(|| std::sync::Mutex::new(Progress::default()))
    }

    /// The window handle, readable from the worker thread -- `S` is a
    /// `thread_local!` and is `None` anywhere else, which in the app was
    /// exactly the bug that silently discarded every generated token.
    fn main_window() -> &'static std::sync::atomic::AtomicUsize {
        static H: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        &H
    }

    fn wake() {
        let h = main_window().load(std::sync::atomic::Ordering::SeqCst);
        if h != 0 {
            unsafe {
                PostMessageW(h as HWND, WM_APP_TICK, 0, 0);
            }
        }
    }

    fn set_status(msg: &str) {
        S.with(|s| {
            if let Some(s) = s.borrow_mut().as_mut() {
                s.status = msg.to_string();
            }
        });
        // Outside the borrow: repainting asks the parent for colours, which
        // borrows, and a double borrow under `panic = "abort"` is silent death.
        let h = main_window().load(std::sync::atomic::Ordering::SeqCst) as HWND;
        if !h.is_null() {
            unsafe {
                InvalidateRect(h, std::ptr::null(), 1);
                UpdateWindow(h);
            }
        }
    }

    pub fn run() {
        unsafe {
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("ChaosSetupWindow");
            let t = theme::SETUP;
            let ground = CreateSolidBrush(t.bg);
            let wc = WNDCLASSW {
                style: 0,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinst,
                hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW as *const u16),
                // Null: `WM_ERASEBKGND` is answered instead, so Windows never
                // paints a ground we are about to paint over.
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
            };
            if RegisterClassW(&wc) == 0 {
                return;
            }
            let title = wide("Chaos Setup");
            // No WS_THICKFRAME or maximise: the layout is fixed, and a resizable
            // window that does not reflow is worse than one that cannot resize.
            // WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, plus
            // WS_CLIPCHILDREN so the step list's repaints do not flash the
            // buttons and the path box underneath them.
            let style = 0x00CA_0000u32 | WS_CLIPCHILDREN;
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                style,
                200,
                120,
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
            // The ground is navy, so the title bar has to be dark or the window
            // wears a white cap.
            let on: i32 = 1;
            DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                &on as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u32,
            );
            set_window_icon(hwnd, hinst);
            main_window().store(hwnd as usize, std::sync::atomic::Ordering::SeqCst);

            let mkfont = |px: i32, weight: i32, face: &str| {
                CreateFontW(
                    px,
                    0,
                    0,
                    0,
                    weight,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    5,
                    0,
                    wide(face).as_ptr(),
                )
            };
            // **Ask what GDI actually gave us.** `CreateFontW` never fails: a
            // face that is not installed is silently substituted, so a display
            // serif chosen for the wordmark can quietly become the UI font.
            let probe = GetDC(hwnd);
            let display_face = first_available_face(probe, theme::FACE_DISPLAY)
                .unwrap_or_else(|| "Times New Roman".to_string());
            ReleaseDC(hwnd, probe);

            let display = mkfont(-72, theme::weight::BOLD, &display_face);
            let font = mkfont(theme::size::BODY, theme::weight::REGULAR, theme::FACE_UI);
            let bold = mkfont(theme::size::HEADING, theme::weight::MEDIUM, theme::FACE_UI);
            let small = mkfont(theme::size::SMALL, theme::weight::REGULAR, theme::FACE_UI);
            let mono = mkfont(theme::size::MONO, theme::weight::REGULAR, theme::FACE_MONO);

            let prefix_text = wide(&default_prefix().to_string_lossy());
            let prefix = CreateWindowExW(
                0,
                wide("EDIT").as_ptr(),
                prefix_text.as_ptr(),
                WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                W / 2 - 230,
                H - 152,
                460,
                26,
                hwnd,
                ID_PREFIX as HMENU,
                hinst,
                std::ptr::null_mut(),
            );
            let mk = |label: &str, id: i32, x: i32, y: i32, w: i32| {
                CreateWindowExW(
                    0,
                    wide("BUTTON").as_ptr(),
                    wide(label).as_ptr(),
                    WS_CHILD | WS_VISIBLE | BS_OWNERDRAW | WS_TABSTOP,
                    x,
                    y,
                    w,
                    42,
                    hwnd,
                    id as HMENU,
                    hinst,
                    std::ptr::null_mut(),
                )
            };
            // One primary action, centred, exactly as Hermes' first frame has
            // it. UNINSTALL is quiet and off to the side: it is not what
            // somebody who just downloaded this came to do.
            let install = mk("INSTALL", ID_INSTALL, W / 2 - 110, 380, 220);
            let uninstall = mk("UNINSTALL", ID_UNINSTALL, W - PAD - 150, H - 96, 150);

            SendMessageW(prefix, WM_SETFONT, mono as WPARAM, 1);
            for h in [install, uninstall] {
                SendMessageW(h, WM_SETFONT, font as WPARAM, 1);
            }
            SendMessageW(
                prefix,
                EM_SETMARGINS,
                EC_LEFTMARGIN | EC_RIGHTMARGIN,
                (8 | (8 << 16)) as LPARAM,
            );

            S.with(|s| {
                *s.borrow_mut() = Some(S {
                    main: hwnd,
                    prefix,
                    screen: Screen::Welcome,
                    theme: t,
                    display,
                    font,
                    bold,
                    small,
                    mono,
                    ground,
                    status: String::new(),
                    done: false,
                })
            });

            if payload::FILES.is_empty() {
                // Said on the face of the window rather than discovered by
                // pressing INSTALL and being told nothing happened.
                set_status("This installer was built with no payload.");
            }

            ShowWindow(hwnd, SW_SHOW);
            UpdateWindow(hwnd);
            let mut msg = MSG::default();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    /// Show or hide the controls that belong to the current screen.
    fn sync_screen() {
        let (screen, install, uninstall, prefix) = S.with(|s| {
            let b = s.borrow();
            match b.as_ref() {
                Some(s) => (
                    s.screen,
                    unsafe { GetDlgItem(s.main, ID_INSTALL) },
                    unsafe { GetDlgItem(s.main, ID_UNINSTALL) },
                    s.prefix,
                ),
                None => (
                    Screen::Welcome,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
            }
        });
        // Borrow closed before Windows is touched: `ShowWindow` repaints, which
        // asks the parent for colours, which borrows.
        let welcome = screen == Screen::Welcome;
        unsafe {
            for h in [install, uninstall, prefix] {
                if !h.is_null() {
                    ShowWindow(h, if welcome { SW_SHOW } else { SW_HIDE });
                }
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
        S.with(|s| {
            if let Some(s) = s.borrow_mut().as_mut() {
                s.screen = Screen::Working;
            }
        });
        *progress().lock().unwrap() = plan(&prefix);
        // The welcome screen's controls belong to the welcome screen. Without
        // this they float over the step list, and the INSTALL button sits in
        // the middle of the very list it started.
        sync_screen();
        unsafe {
            InvalidateRect(
                main_window().load(std::sync::atomic::Ordering::SeqCst) as HWND,
                std::ptr::null(),
                1,
            );
        }
        // On a worker, so the window keeps painting. An installer that freezes
        // for the duration of its own work looks like one that has hung, and
        // this one had nothing to say while it ran.
        std::thread::spawn(move || {
            perform(&prefix);
            wake();
        });
    }

    /// The steps this install will take, before any of them run.
    ///
    /// Built up front so the list is complete on screen from the first frame --
    /// a progress list that grows as it goes cannot show how far along it is.
    fn plan(prefix: &std::path::Path) -> Progress {
        let mut steps = vec![
            Step::new("Checking for an existing install"),
            Step::new("Creating the program folder"),
            Step::new("Creating the models folder"),
            Step::new("Removing files this version replaces"),
        ];
        for f in payload::FILES.iter() {
            steps.push(Step::new(format!("Writing {}", f.name)));
        }
        steps.push(Step::new("Recording the file manifest"));
        steps.push(Step::new("Adding Chaos to your PATH"));
        steps.push(Step::new("Creating the Start Menu entry"));
        steps.push(Step::new("Registering with Add or remove programs"));
        let _ = prefix;
        Progress {
            steps,
            ..Progress::default()
        }
    }

    /// Append a line to `%TEMP%\chaos-setup.log`.
    ///
    /// **An installer should say what it did somewhere durable.** This is also
    /// the only way to find out where one died: the first stepped build aborted
    /// mid-install with no panic message and no window, and the Windows event
    /// log gave a fastfail code and nothing else. Opened and closed per line so
    /// a hard abort cannot lose a buffered write.
    fn log(line: &str) {
        use std::io::Write;
        let p = std::env::temp_dir().join("chaos-setup.log");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
        {
            let _ = writeln!(f, "{line}");
        }
    }

    /// Advance one step, timing it. `f` returns an error message on failure.
    ///
    /// Every step reports before and after, so the window can show which line
    /// is in flight rather than only which have finished.
    fn step(i: usize, f: impl FnOnce() -> Result<(), String>) -> bool {
        {
            let mut p = progress().lock().unwrap();
            if p.failure.is_some() {
                return false;
            }
            match p.steps.get_mut(i) {
                Some(s) => {
                    log(&format!("[{i}] start {}", s.label));
                    s.state = StepState::Running;
                }
                // A step index with no step is a bug in `plan`, and silently
                // doing the work anyway would leave the bar stuck.
                None => {
                    log(&format!("[{i}] NO SUCH STEP -- plan and perform disagree"));
                    p.failure = Some(format!("internal: step {i} is not in the plan"));
                    return false;
                }
            }
        }
        wake();
        let started = Instant::now();
        let outcome = f();
        let ms = started.elapsed().as_millis() as u64;
        {
            let mut p = progress().lock().unwrap();
            if let Some(s) = p.steps.get_mut(i) {
                s.millis = ms;
                s.state = if outcome.is_ok() {
                    StepState::Done
                } else {
                    StepState::Failed
                };
            }
            if let Err(e) = &outcome {
                p.failure = Some(e.clone());
            }
        }
        match &outcome {
            Ok(()) => log(&format!("[{i}] done in {ms}ms")),
            Err(e) => log(&format!("[{i}] FAILED after {ms}ms: {e}")),
        }
        wake();
        outcome.is_ok()
    }

    /// Do the install, one reported step at a time.
    fn perform(prefix: &std::path::Path) {
        log(&format!(
            "--- chaos-setup {} installing to {}",
            env!("CARGO_PKG_VERSION"),
            prefix.display()
        ));
        if payload::FILES.is_empty() {
            let mut p = progress().lock().unwrap();
            p.failure = Some("this build carries no payload".into());
            return;
        }
        let bin = bin_dir(prefix);
        let models = default_models_dir();
        let incoming: Vec<String> = payload::FILES.iter().map(|f| f.name.to_string()).collect();
        let mut before = None;
        let mut i = 0;

        if !step(i, || {
            // Read before a byte is written: afterwards it would describe this
            // install rather than the one being replaced.
            before = existing_install(prefix);
            Ok(())
        }) {
            return;
        }
        i += 1;
        if !step(i, || {
            std::fs::create_dir_all(&bin)
                .map_err(|e| format!("cannot create {}: {e}", bin.display()))
        }) {
            return;
        }
        i += 1;
        if !step(i, || {
            std::fs::create_dir_all(&models)
                .map_err(|e| format!("cannot create {}: {e}", models.display()))
        }) {
            return;
        }
        i += 1;
        if !step(i, || {
            // An upgrade must not leave a binary this version dropped sitting
            // on PATH, still runnable and now wrong.
            if let Ok(prev) = std::fs::read_to_string(manifest_path(prefix)) {
                let prev: Vec<String> = prev
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect();
                for old in stale(&prev, &incoming) {
                    let _ = std::fs::remove_file(bin.join(old));
                }
            }
            Ok(())
        }) {
            return;
        }

        for f in payload::FILES.iter() {
            i += 1;
            let dest = bin.join(f.name);
            if !step(i, || {
                std::fs::write(&dest, f.bytes).map_err(|e| {
                    // A running binary is locked, and this is the one failure a
                    // user hits twice: install, run, install again.
                    format!(
                        "cannot write {} ({e}). Close Chaos and run this again.",
                        f.name
                    )
                })
            }) {
                return;
            }
        }

        i += 1;
        if !step(i, || {
            std::fs::write(manifest_path(prefix), incoming.join("\n"))
                .map_err(|e| format!("cannot record the manifest: {e}"))?;
            // Recorded so the NEXT installer can say what it is replacing.
            let _ = std::fs::write(version_path(prefix), env!("CARGO_PKG_VERSION"));
            Ok(())
        }) {
            return;
        }
        i += 1;
        let bin_s = bin.to_string_lossy().to_string();
        if !step(i, || {
            add_to_path(&bin_s);
            Ok(())
        }) {
            return;
        }
        i += 1;
        if !step(i, || {
            make_shortcut(&bin);
            Ok(())
        }) {
            return;
        }
        i += 1;
        if !step(i, || {
            register_uninstall(prefix);
            Ok(())
        }) {
            return;
        }

        log("all steps complete");
        let head = upgrade_line(before.as_ref(), env!("CARGO_PKG_VERSION"));
        let mut p = progress().lock().unwrap();
        p.report = vec![
            head,
            String::new(),
            format!("{} files installed to", payload::FILES.len()),
            format!("  {}", bin.display()),
            String::new(),
            "Models folder".to_string(),
            format!("  {}", models.display()),
            String::new(),
            "Added to your PATH, with a Start Menu entry.".to_string(),
            "Open a NEW terminal for the PATH change to apply.".to_string(),
        ];
    }

    fn do_uninstall() {
        let prefix = prefix_value();
        let inside = running_inside(&prefix);
        let (ok, msg) = uninstall_to(&prefix);
        set_status(&msg);

        // **When this window is running from inside the folder being removed,
        // it has to go.** The staged helper cannot delete a directory while
        // this process holds an executable open in it, so it retries for ten
        // seconds and then gives up -- which is what "I cannot uninstall the
        // app" was. Show the result, then leave.
        if ok && inside {
            unsafe {
                MessageBoxW(
                    std::ptr::null_mut(),
                    wide(
                        "Chaos has been removed.

                         Your models were left where they are.

                         This window will now close so the last files can be deleted.",
                    )
                    .as_ptr(),
                    wide("Chaos Setup").as_ptr(),
                    MB_ICONINFORMATION,
                );
                PostQuitMessage(0);
            }
        }
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
        // Read before a byte is written: afterwards it describes this install
        // rather than the one being replaced.
        let before = existing_install(&prefix);
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
        // Recorded so the NEXT installer can say what it is replacing.
        let _ = std::fs::write(version_path(&prefix), env!("CARGO_PKG_VERSION"));

        add_to_path(&bin.to_string_lossy());
        make_shortcut(&bin);
        register_uninstall(&prefix);

        let head = upgrade_line(before.as_ref(), env!("CARGO_PKG_VERSION"));
        let nl = char::from(10);
        (
            true,
            [
                head,
                String::new(),
                format!("{} files installed to", payload::FILES.len()),
                format!("  {}", bin.display()),
                String::new(),
                "Models folder".to_string(),
                format!("  {}", models.display()),
                String::new(),
                "Added to your PATH, with a Start Menu entry.".to_string(),
                "Open a NEW terminal for the PATH change to apply.".to_string(),
                String::new(),
                "Next: run chaos-app for the window, or chaos-pull --list for models.".to_string(),
            ]
            .join(&nl.to_string()),
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
        (true, {
            let nl = char::from(10).to_string();
            [
                "Chaos has been removed.".to_string(),
                String::new(),
                "Deleted".to_string(),
                format!("  {}", bin.display()),
                "Also removed: the PATH entry, the Start Menu shortcut and the".to_string(),
                "Add/Remove Programs entry.".to_string(),
                String::new(),
                "KEPT, on purpose".to_string(),
                format!("  {}", default_models_dir().display()),
                "Your downloaded models are untouched. Delete that folder".to_string(),
                "yourself if you want the space back.".to_string(),
            ]
            .join(&nl)
        })
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

    // -- painting ------------------------------------------------------------

    unsafe fn text(hdc: HDC, r: RECT, txt: &str, font: HFONT, colour: u32, flags: u32) {
        let old = SelectObject(hdc, font as HGDIOBJ);
        SetTextColor(hdc, colour);
        SetBkMode(hdc, TRANSPARENT);
        let mut rc = r;
        let w: Vec<u16> = txt.encode_utf16().collect();
        // **Never hand Windows an empty Vec's pointer.** `Vec::as_ptr` on an
        // empty vector returns a dangling (aligned but unallocated) address,
        // and `DrawTextW` dereferences it -- which is what killed the installer
        // the moment its report reached a blank line. There is nothing to draw
        // anyway, so this is a guard rather than a workaround.
        if !w.is_empty() {
            DrawTextW(
                hdc,
                w.as_ptr(),
                w.len() as i32,
                &mut rc,
                flags | DT_NOPREFIX,
            );
        }
        SelectObject(hdc, old);
    }

    unsafe fn fill(hdc: HDC, r: RECT, colour: u32) {
        let b = CreateSolidBrush(colour);
        FillRect(hdc, &r, b);
        DeleteObject(b as HGDIOBJ);
    }

    unsafe fn line(hdc: HDC, x: i32, y: i32, w: i32, colour: u32) {
        fill(
            hdc,
            RECT {
                left: x,
                top: y,
                right: x + w,
                bottom: y + 1,
            },
            colour,
        );
    }

    /// The wordmark, letter by letter.
    ///
    /// Hermes tracks its wordmark at `0.08em`, which is most of why it reads as
    /// a wordmark rather than as a heading. GDI has no letter-spacing, so the
    /// glyphs are placed one at a time and each advance is measured.
    unsafe fn wordmark(hdc: HDC, cx: i32, y: i32, txt: &str, font: HFONT, colour: u32, track: i32) {
        let old = SelectObject(hdc, font as HGDIOBJ);
        SetTextColor(hdc, colour);
        SetBkMode(hdc, TRANSPARENT);
        let glyphs: Vec<Vec<u16>> = txt.chars().map(|c| vec![c as u16]).collect();
        let mut widths = Vec::with_capacity(glyphs.len());
        let mut total = 0;
        for g in &glyphs {
            let mut sz = SIZE::default();
            GetTextExtentPoint32W(hdc, g.as_ptr(), 1, &mut sz);
            widths.push(sz.cx);
            total += sz.cx + track;
        }
        total = (total - track).max(0);
        let mut x = cx - total / 2;
        for (g, w) in glyphs.iter().zip(&widths) {
            TextOutW(hdc, x, y, g.as_ptr(), 1);
            x += w + track;
        }
        SelectObject(hdc, old);
    }

    /// The mark, box-filtered and blended -- the same routine the app uses, for
    /// the same reason: the 56px terminal bitmap stretched to 72 was a blob.
    unsafe fn mark(hdc: HDC, x: i32, y: i32, px: usize, ink: u32, ground: u32) {
        let cov = chaos_app::art::logo_scaled(px);
        let chan = |c: u32, shift: u32| ((c >> shift) & 0xFF) as i32;
        let mut buf = vec![0u8; px * px * 4];
        for row in 0..px {
            for col in 0..px {
                let a = i32::from(cov[(px - 1 - row) * px + col]);
                let i = (row * px + col) * 4;
                for (o, shift) in [(0usize, 16u32), (1, 8), (2, 0)] {
                    let (f, b) = (chan(ink, shift), chan(ground, shift));
                    buf[i + o] = (b + (f - b) * a / 255) as u8;
                }
            }
        }
        let bmi = BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: px as i32,
            biHeight: px as i32,
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
            x,
            y,
            px as i32,
            px as i32,
            0,
            0,
            px as i32,
            px as i32,
            buf.as_ptr() as *const _,
            &bmi,
            DIB_RGB_COLORS,
            SRCCOPY,
        );
    }

    unsafe fn paint(hwnd: HWND) {
        let mut ps: PAINTSTRUCT = std::mem::zeroed();
        let hdc = BeginPaint(hwnd, &mut ps);
        let mut r = RECT::default();
        GetClientRect(hwnd, &mut r);

        // Double-buffered: the step list repaints on every step transition.
        let mem = CreateCompatibleDC(hdc);
        let bmp = CreateCompatibleBitmap(hdc, r.right.max(1), r.bottom.max(1));
        let old_bmp = SelectObject(mem, bmp);

        S.with(|s| {
            let b = s.borrow();
            let Some(s) = b.as_ref() else { return };
            fill(mem, r, s.theme.bg);
            match s.screen {
                Screen::Welcome => paint_welcome(mem, s, r),
                Screen::Working => paint_working(mem, s, r),
                Screen::Report => paint_report(mem, s, r),
            }
        });

        BitBlt(hdc, 0, 0, r.right, r.bottom, mem, 0, 0, SRCCOPY);
        SelectObject(mem, old_bmp);
        DeleteObject(bmp);
        DeleteDC(mem);
        EndPaint(hwnd, &ps);
    }

    /// One idea, one sentence, one action. Nothing else is on it.
    unsafe fn paint_welcome(hdc: HDC, s: &S, r: RECT) {
        let t = &s.theme;
        let cx = r.right / 2;
        mark(hdc, cx - 34, 84, 68, t.fg, t.bg);
        wordmark(hdc, cx, 186, "CHAOS", s.display, t.fg, 10);
        text(
            hdc,
            RECT {
                left: PAD * 2,
                top: 300,
                right: r.right - PAD * 2,
                bottom: 352,
            },
            "Run models that do not fit in your memory. This sets everything up \
             here -- it takes a few seconds.",
            s.font,
            t.fg_secondary,
            DT_CENTER | DT_WORDBREAK,
        );
        text(
            hdc,
            RECT {
                left: PAD,
                top: r.bottom - 138,
                right: r.right - PAD,
                bottom: r.bottom - 118,
            },
            "Installs to",
            s.small,
            t.fg_tertiary,
            DT_CENTER | DT_SINGLELINE,
        );
        text(
            hdc,
            RECT {
                left: PAD,
                top: r.bottom - 44,
                right: r.right - PAD,
                bottom: r.bottom - 22,
            },
            "Per-user. No administrator rights needed.",
            s.small,
            t.fg_tertiary,
            DT_CENTER | DT_SINGLELINE,
        );
    }

    /// The named step list: what is happening, what has happened, and how long
    /// each took. This is the difference between "it is working" and "it has
    /// frozen", and the old installer had no way to tell you which.
    unsafe fn paint_working(hdc: HDC, s: &S, r: RECT) {
        let t = &s.theme;
        let x = PAD;
        let w = r.right - PAD * 2;
        mark(hdc, x, PAD, 44, t.fg, t.bg);
        text(
            hdc,
            RECT {
                left: x + 62,
                top: PAD,
                right: r.right - PAD,
                bottom: PAD + 30,
            },
            "Setting up Chaos",
            s.bold,
            t.fg,
            DT_LEFT | DT_SINGLELINE,
        );
        text(
            hdc,
            RECT {
                left: x + 62,
                top: PAD + 26,
                right: r.right - PAD,
                bottom: PAD + 68,
            },
            "A one-time setup. Chaos is unpacking its binaries and configuring \
             this machine. Later launches skip this.",
            s.small,
            t.fg_secondary,
            DT_LEFT | DT_WORDBREAK,
        );

        let p = progress().lock().unwrap();
        let total = p.steps.len();
        let done = p.done_count();
        let pct = p.percent();
        let bar_y = PAD + 98;
        text(
            hdc,
            RECT {
                left: x,
                top: bar_y - 24,
                right: x + w / 2,
                bottom: bar_y - 2,
            },
            &format!("{done} of {total} steps complete"),
            s.small,
            t.fg_secondary,
            DT_LEFT | DT_SINGLELINE,
        );
        text(
            hdc,
            RECT {
                left: x + w / 2,
                top: bar_y - 24,
                right: x + w,
                bottom: bar_y - 2,
            },
            &format!("{pct}%"),
            s.small,
            t.fg_secondary,
            DT_RIGHT | DT_SINGLELINE,
        );
        fill(
            hdc,
            RECT {
                left: x,
                top: bar_y,
                right: x + w,
                bottom: bar_y + 4,
            },
            t.stroke_3,
        );
        fill(
            hdc,
            RECT {
                left: x,
                top: bar_y,
                right: x + w * pct as i32 / 100,
                bottom: bar_y + 4,
            },
            if p.failure.is_some() { t.red } else { t.accent },
        );

        // Only as many rows as fit -- and when the list is longer, scroll so the
        // running step stays on screen rather than being silently hidden.
        let row_h = 30;
        let top = bar_y + 28;
        let rows = (((r.bottom - 64 - top) / row_h).max(1)) as usize;
        let focus = p.running().unwrap_or(done);
        let first = if focus >= rows { focus + 1 - rows } else { 0 };
        for (n, st) in p.steps.iter().enumerate().skip(first).take(rows) {
            let y = top + (n - first) as i32 * row_h;
            let (colour, font) = match st.state {
                StepState::Done => (t.fg_secondary, s.font),
                StepState::Running => (t.fg, s.bold),
                StepState::Failed => (t.red, s.bold),
                StepState::Waiting => (t.fg_tertiary, s.font),
            };
            // Markers are drawn, not typed: a tick from a font at this size is
            // whatever that font happens to have, and it is rarely centred.
            match st.state {
                StepState::Done => {
                    let pen = CreatePen(0, 2, t.accent);
                    let old = SelectObject(hdc, pen);
                    MoveToEx(hdc, x + 2, y + 12, std::ptr::null_mut());
                    LineTo(hdc, x + 6, y + 16);
                    LineTo(hdc, x + 14, y + 7);
                    SelectObject(hdc, old);
                    DeleteObject(pen);
                }
                StepState::Running => fill(
                    hdc,
                    RECT {
                        left: x + 3,
                        top: y + 8,
                        right: x + 12,
                        bottom: y + 17,
                    },
                    t.fg,
                ),
                StepState::Failed => {
                    let pen = CreatePen(0, 2, t.red);
                    let old = SelectObject(hdc, pen);
                    MoveToEx(hdc, x + 3, y + 7, std::ptr::null_mut());
                    LineTo(hdc, x + 14, y + 18);
                    MoveToEx(hdc, x + 14, y + 7, std::ptr::null_mut());
                    LineTo(hdc, x + 3, y + 18);
                    SelectObject(hdc, old);
                    DeleteObject(pen);
                }
                StepState::Waiting => {}
            }
            text(
                hdc,
                RECT {
                    left: x + 28,
                    top: y,
                    right: x + w - 96,
                    bottom: y + row_h,
                },
                &st.label,
                font,
                colour,
                DT_LEFT | DT_SINGLELINE | DT_VCENTER | DT_END_ELLIPSIS,
            );
            if matches!(st.state, StepState::Done | StepState::Failed) {
                text(
                    hdc,
                    RECT {
                        left: x + w - 96,
                        top: y,
                        right: x + w,
                        bottom: y + row_h,
                    },
                    &human_millis(st.millis),
                    s.mono,
                    t.fg_tertiary,
                    DT_RIGHT | DT_SINGLELINE | DT_VCENTER,
                );
            }
        }
    }

    /// The report. **The window does not close on its own**: an installer that
    /// vanishes has not told you whether it worked.
    unsafe fn paint_report(hdc: HDC, s: &S, r: RECT) {
        let t = &s.theme;
        let x = PAD;
        let w = r.right - PAD * 2;
        let failed = progress().lock().unwrap().failure.clone();
        mark(hdc, x, PAD, 44, t.fg, t.bg);
        text(
            hdc,
            RECT {
                left: x + 62,
                top: PAD + 4,
                right: r.right - PAD,
                bottom: PAD + 38,
            },
            if failed.is_some() {
                "Setup did not finish"
            } else {
                "Chaos is installed"
            },
            s.bold,
            if failed.is_some() { t.red } else { t.fg },
            DT_LEFT | DT_SINGLELINE,
        );
        line(hdc, x, PAD + 66, w, t.stroke_3);

        let mut y = PAD + 88;
        if let Some(why) = &failed {
            text(
                hdc,
                RECT {
                    left: x,
                    top: y,
                    right: x + w,
                    bottom: y + 64,
                },
                why,
                s.font,
                t.fg,
                DT_LEFT | DT_WORDBREAK,
            );
            y += 74;
        }
        for l in s.status.split(char::from(10)) {
            let path = l.starts_with("  ");
            text(
                hdc,
                RECT {
                    left: x,
                    top: y,
                    right: x + w,
                    bottom: y + 22,
                },
                l,
                if path { s.mono } else { s.font },
                if path { t.fg } else { t.fg_secondary },
                DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS,
            );
            y += 21;
        }
        if failed.is_none() {
            line(hdc, x, r.bottom - 132, w, t.stroke_3);
            text(
                hdc,
                RECT {
                    left: x,
                    top: r.bottom - 120,
                    right: x + w,
                    bottom: r.bottom - 98,
                },
                "chaos-app  the window        chaos-run  the command line",
                s.mono,
                t.fg_tertiary,
                DT_LEFT | DT_SINGLELINE,
            );
        }
    }

    /// The buttons, drawn as Hermes draws its `[ INSTALL ]`: a hairline box, a
    /// tracked label, no fill.
    unsafe fn draw_item(di: &DRAWITEMSTRUCT) {
        let selected = di.itemState & ODS_SELECTED != 0;
        let disabled = di.itemState & ODS_DISABLED != 0;
        let snapshot = S.with(|s| s.borrow().as_ref().map(|s| (s.theme, s.font, s.bold)));
        let Some((t, font, bold)) = snapshot else {
            return;
        };
        let primary = di.CtlID as i32 == ID_INSTALL;
        let r = di.rcItem;

        fill(di.hDC, r, if selected { t.soft_active } else { t.bg });
        if !disabled {
            let colour = if primary { t.accent } else { t.stroke_2 };
            for q in [
                RECT {
                    bottom: r.top + 1,
                    ..r
                },
                RECT {
                    top: r.bottom - 1,
                    ..r
                },
                RECT {
                    right: r.left + 1,
                    ..r
                },
                RECT {
                    left: r.right - 1,
                    ..r
                },
            ] {
                fill(di.hDC, q, colour);
            }
        }
        let n = GetWindowTextLengthW(di.hwndItem);
        if n <= 0 {
            return;
        }
        let mut buf = vec![0u16; n as usize + 1];
        let got = GetWindowTextW(di.hwndItem, buf.as_mut_ptr(), n + 1);
        let label = String::from_utf16_lossy(&buf[..got.max(0) as usize]);
        wordmark(
            di.hDC,
            (r.left + r.right) / 2,
            (r.top + r.bottom) / 2 - 9,
            &label,
            if primary { bold } else { font },
            if disabled {
                t.fg_tertiary
            } else if primary {
                t.accent
            } else {
                t.fg_secondary
            },
            3,
        );
    }

    /// Put the embedded icon on the window itself.
    ///
    /// The resource compiled into the executable is what Explorer shows for the
    /// *file*. The title bar, the taskbar button and the Alt-Tab strip read the
    /// *window's* icon, which is unset until something sends `WM_SETICON` --
    /// which is why an executable can have a perfectly good icon in Explorer and
    /// still show a blank page while it is running.
    ///
    /// Both sizes: small is the title bar, big is the taskbar. Resource id 1,
    /// matching what `build.rs` writes into the `.rc`.
    unsafe fn set_window_icon(hwnd: HWND, hinst: HINSTANCE) {
        let id = 1u16 as *const u16;
        let big = LoadImageW(hinst, id, IMAGE_ICON, 0, 0, LR_DEFAULTSIZE | LR_SHARED);
        if !big.is_null() {
            SendMessageW(hwnd, WM_SETICON, ICON_BIG, big as LPARAM);
        }
        let small = LoadImageW(hinst, id, IMAGE_ICON, 16, 16, LR_SHARED);
        if !small.is_null() {
            SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small as LPARAM);
        }
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
            WM_ERASEBKGND => 1,
            WM_APP_TICK => {
                // The worker moved a step. Repaint, and when the run is over
                // switch to the report -- the window never closes itself.
                let finished = progress().lock().unwrap().finished();
                if finished {
                    let report = {
                        let p = progress().lock().unwrap();
                        p.report.join(&char::from(10).to_string())
                    };
                    S.with(|s| {
                        if let Some(s) = s.borrow_mut().as_mut() {
                            s.screen = Screen::Report;
                            s.done = true;
                        }
                    });
                    set_status(&report);
                    sync_screen();
                }
                InvalidateRect(hwnd, std::ptr::null(), 1);
                0
            }
            WM_CTLCOLOREDIT | WM_CTLCOLORSTATIC | WM_CTLCOLORBTN => {
                let (fg, bg, brush) = S.with(|s| {
                    let b = s.borrow();
                    match b.as_ref() {
                        Some(s) => (s.theme.fg, s.theme.soft, s.ground),
                        None => (0, 0, std::ptr::null_mut()),
                    }
                });
                SetTextColor(wp as HDC, fg);
                SetBkColor(wp as HDC, bg);
                // A control fill one shade off the ground, so the path box reads
                // as a box without needing a border on a navy field.
                brush as LRESULT
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
