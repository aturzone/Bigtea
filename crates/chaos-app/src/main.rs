//! Chaos, as a window.
//!
//! A real Win32 application: `RegisterClassW`, a message loop, native controls,
//! GDI painting. No browser, no webview, no HTML, and no GUI dependency -- the
//! whole surface it uses is declared in `win32.rs` against libraries Windows
//! already ships.
//!
//! **Two colours, and only two.** `#000000` and `#FFFFFF`. Every control is
//! repainted to match, because the defaults are grey and a two-value design
//! with one grey panel in it is not a two-value design. Emphasis is carried by
//! inversion and by rules, which is what you do when you have no third value --
//! and it is what Atur asked for.
//!
//! The engine runs as a child `chaos-serve` process; `client.rs` says why at
//! length. The short version is that unloading a model has to actually free
//! 7 GiB, and that a second in-process construction path is where this codebase
//! has historically hidden its worst bug.

#![cfg_attr(not(windows), allow(dead_code))]
// The window subsystem, so double-clicking it does not also open a console.
// `main` still runs; only the console allocation is suppressed.
#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(windows)]
mod win32;

#[cfg(not(windows))]
fn main() {
    eprintln!(
        "chaos-app is the Windows application; on this platform use chaos-run or chaos-serve."
    );
    std::process::exit(1);
}

#[cfg(windows)]
fn main() {
    windows_app::run();
}

#[cfg(windows)]
mod windows_app {
    use crate::win32::*;
    use chaos_app::{art, client, models};
    use std::cell::RefCell;
    use std::process::{Child, Command};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    const ID_LIST: i32 = 101;
    const ID_LOAD: i32 = 102;
    const ID_UNLOAD: i32 = 103;
    const ID_SEND: i32 = 104;
    const ID_OUT: i32 = 105;
    const ID_IN: i32 = 106;
    const ID_CACHE: i32 = 107;
    const ID_THREADS: i32 = 108;
    const ID_REFRESH: i32 = 109;

    const SIDEBAR: i32 = 300;
    const PAD: i32 = 14;
    const LOGO_PX: i32 = 96;

    /// State the worker thread and the window both touch.
    #[derive(Default)]
    struct Shared {
        /// Text produced since the UI last drained it.
        pending: String,
        /// Set when the answer is complete, so the UI can re-enable Send.
        finished: bool,
        tokens: u32,
        started: Option<std::time::Instant>,
        status: String,
    }

    struct Ui {
        main: HWND,
        list: HWND,
        out: HWND,
        input: HWND,
        load: HWND,
        unload: HWND,
        send: HWND,
        cache: HWND,
        threads: HWND,
        font: HFONT,
        mono: HFONT,
        black: HBRUSH,
        entries: Vec<models::Entry>,
        server: Option<Child>,
        port: u16,
        history: Vec<(String, String)>,
        answer: String,
    }

    thread_local! {
        static UI: RefCell<Option<Ui>> = const { RefCell::new(None) };
    }

    fn shared() -> &'static Mutex<Shared> {
        static S: std::sync::OnceLock<Mutex<Shared>> = std::sync::OnceLock::new();
        S.get_or_init(|| Mutex::new(Shared::default()))
    }

    fn busy() -> &'static AtomicBool {
        static B: AtomicBool = AtomicBool::new(false);
        &B
    }

    pub fn run() {
        unsafe {
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("ChaosAppWindow");
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

            let title = wide("Chaos");
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                WS_OVERLAPPEDWINDOW,
                120,
                80,
                1160,
                760,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinst,
                std::ptr::null_mut(),
            );
            if hwnd.is_null() {
                return;
            }

            // The compositor draws the title bar, so it stays light unless
            // asked. Ignored on Windows builds that predate the attribute.
            let on: i32 = 1;
            DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                &on as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u32,
            );

            build_controls(hwnd, hinst, black);
            ShowWindow(hwnd, SW_SHOW);
            UpdateWindow(hwnd);

            let mut msg = MSG::default();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    unsafe fn make_font(px: i32, weight: i32, face: &str) -> HFONT {
        let name = wide(face);
        CreateFontW(px, 0, 0, 0, weight, 0, 0, 0, 1, 0, 0, 5, 0, name.as_ptr())
    }

    unsafe fn child(
        parent: HWND,
        class: &str,
        text: &str,
        style: u32,
        id: i32,
        hinst: HINSTANCE,
    ) -> HWND {
        let c = wide(class);
        let t = wide(text);
        CreateWindowExW(
            0,
            c.as_ptr(),
            t.as_ptr(),
            WS_CHILD | WS_VISIBLE | style,
            0,
            0,
            10,
            10,
            parent,
            id as HMENU,
            hinst,
            std::ptr::null_mut(),
        )
    }

    unsafe fn build_controls(hwnd: HWND, hinst: HINSTANCE, black: HBRUSH) {
        let font = make_font(-15, 400, "Segoe UI");
        let mono = make_font(-14, 400, "Consolas");

        let list = child(
            hwnd,
            "LISTBOX",
            "",
            LBS_NOTIFY | LBS_OWNERDRAWFIXED | LBS_HASSTRINGS | WS_VSCROLL | WS_BORDER,
            ID_LIST,
            hinst,
        );
        let out = child(
            hwnd,
            "EDIT",
            "",
            ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL | WS_BORDER,
            ID_OUT,
            hinst,
        );
        let input = child(
            hwnd,
            "EDIT",
            "",
            ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN | WS_BORDER | WS_TABSTOP,
            ID_IN,
            hinst,
        );
        let load = child(
            hwnd,
            "BUTTON",
            "LOAD",
            BS_OWNERDRAW | WS_TABSTOP,
            ID_LOAD,
            hinst,
        );
        let unload = child(
            hwnd,
            "BUTTON",
            "UNLOAD",
            BS_OWNERDRAW | WS_TABSTOP,
            ID_UNLOAD,
            hinst,
        );
        let refresh = child(
            hwnd,
            "BUTTON",
            "RESCAN",
            BS_OWNERDRAW | WS_TABSTOP,
            ID_REFRESH,
            hinst,
        );
        let send = child(
            hwnd,
            "BUTTON",
            "SEND",
            BS_OWNERDRAW | WS_TABSTOP,
            ID_SEND,
            hinst,
        );
        let cache = child(hwnd, "EDIT", "", WS_BORDER | WS_TABSTOP, ID_CACHE, hinst);
        let threads = child(hwnd, "EDIT", "", WS_BORDER | WS_TABSTOP, ID_THREADS, hinst);

        for h in [
            list, out, input, load, unload, refresh, send, cache, threads,
        ] {
            SendMessageW(h, WM_SETFONT, font as WPARAM, 1);
        }
        SendMessageW(out, WM_SETFONT, mono as WPARAM, 1);
        SendMessageW(input, WM_SETFONT, mono as WPARAM, 1);

        // An owner-draw list uses a fixed row height that defaults to
        // roughly the system font's, which clipped the model name in half.
        SendMessageW(list, LB_SETITEMHEIGHT, 0, 26);

        EnableWindow(unload, 0);
        EnableWindow(send, 0);

        UI.with(|u| {
            *u.borrow_mut() = Some(Ui {
                main: hwnd,
                list,
                out,
                input,
                load,
                unload,
                send,
                cache,
                threads,
                font,
                mono,
                black,
                entries: Vec::new(),
                server: None,
                port: 8231,
                history: Vec::new(),
                answer: String::new(),
            })
        });
        rescan();
    }

    /// Re-read the models directory into the list.
    fn rescan() {
        UI.with(|u| {
            let mut b = u.borrow_mut();
            let Some(ui) = b.as_mut() else { return };
            unsafe {
                SendMessageW(ui.list, LB_RESETCONTENT, 0, 0);
            }
            ui.entries = models::list();
            if ui.entries.is_empty() {
                let t = wide("no models found -- put a .gguf in ~/.chaos/models");
                unsafe {
                    SendMessageW(ui.list, LB_ADDSTRING, 0, t.as_ptr() as LPARAM);
                }
            } else {
                for e in &ui.entries {
                    let t = wide(&models::row(e));
                    unsafe {
                        SendMessageW(ui.list, LB_ADDSTRING, 0, t.as_ptr() as LPARAM);
                    }
                }
                unsafe {
                    SendMessageW(ui.list, LB_SETCURSEL, 0, 0);
                }
            }
        });
    }

    fn set_status(text: &str) {
        shared().lock().unwrap().status = text.to_string();
        UI.with(|u| {
            if let Some(ui) = u.borrow().as_ref() {
                unsafe {
                    InvalidateRect(ui.main, std::ptr::null(), 1);
                }
            }
        });
    }

    fn append_out(text: &str) {
        UI.with(|u| {
            let b = u.borrow();
            let Some(ui) = b.as_ref() else { return };
            let w = wide(text);
            unsafe {
                // Move the caret to the end, then replace an empty selection:
                // the only way to append to an EDIT without re-sending the
                // whole buffer, which for a long answer is quadratic.
                let len = GetWindowTextLengthW(ui.out);
                SendMessageW(ui.out, EM_SETSEL, len as WPARAM, len as LPARAM);
                SendMessageW(ui.out, EM_REPLACESEL, 0, w.as_ptr() as LPARAM);
                SendMessageW(ui.out, EM_SCROLLCARET, 0, 0);
            }
        });
    }

    fn control_text(h: HWND) -> String {
        unsafe {
            let n = GetWindowTextLengthW(h);
            if n <= 0 {
                return String::new();
            }
            let mut buf = vec![0u16; n as usize + 1];
            let got = GetWindowTextW(h, buf.as_mut_ptr(), n + 1);
            String::from_utf16_lossy(&buf[..got.max(0) as usize])
        }
    }

    /// Start `chaos-serve` on the selected model, hidden.
    fn load_model() {
        let (path, cache, threads, port) = UI.with(|u| {
            let b = u.borrow();
            let Some(ui) = b.as_ref() else {
                return (None, String::new(), String::new(), 0u16);
            };
            let sel = unsafe { SendMessageW(ui.list, LB_GETCURSEL, 0, 0) };
            let path = (sel >= 0)
                .then(|| ui.entries.get(sel as usize))
                .flatten()
                .map(|e| e.path.clone());
            (
                path,
                control_text(ui.cache),
                control_text(ui.threads),
                ui.port,
            )
        });
        let Some(path) = path else {
            set_status("select a model first");
            return;
        };

        // Next to us, not on PATH: an app that finds a different Chaos than the
        // one it shipped with is a support problem nobody can reproduce.
        let exe = match std::env::current_exe() {
            Ok(p) => p.with_file_name("chaos-serve.exe"),
            Err(e) => {
                set_status(&format!("cannot locate chaos-serve: {e}"));
                return;
            }
        };
        if !exe.exists() {
            set_status("chaos-serve.exe is missing from this folder");
            return;
        }

        let mut cmd = Command::new(&exe);
        cmd.arg(&path).arg("--port").arg(port.to_string());
        if let Ok(v) = cache.trim().parse::<f64>() {
            if v > 0.0 {
                cmd.arg("--cache").arg(format!("{v}"));
            }
        }
        if let Ok(v) = threads.trim().parse::<u32>() {
            if v > 0 {
                cmd.arg("-t").arg(v.to_string());
            }
        }
        // Unverified architectures refuse by name. The app is for running what
        // you have, and the refusal is already explained in the status line.
        cmd.arg("--force");
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        match cmd.spawn() {
            Ok(c) => {
                UI.with(|u| {
                    if let Some(ui) = u.borrow_mut().as_mut() {
                        ui.server = Some(c);
                        ui.history.clear();
                        unsafe {
                            EnableWindow(ui.load, 0);
                            EnableWindow(ui.unload, 1);
                            SetWindowTextW(ui.out, wide("").as_ptr());
                        }
                    }
                });
                set_status("loading -- a large model takes a while");
                let port2 = port;
                std::thread::spawn(move || {
                    // Poll rather than parse stdout: readiness is exactly "does
                    // it answer", and that is the same check a client makes.
                    for _ in 0..1200 {
                        if client::health(port2) {
                            let mut s = shared().lock().unwrap();
                            s.status = "ready".into();
                            s.finished = true;
                            drop(s);
                            notify();
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                    let mut s = shared().lock().unwrap();
                    s.status = "the model did not come up".into();
                    s.finished = true;
                    drop(s);
                    notify();
                });
            }
            Err(e) => set_status(&format!("could not start: {e}")),
        }
    }

    fn unload_model() {
        UI.with(|u| {
            let mut b = u.borrow_mut();
            let Some(ui) = b.as_mut() else { return };
            if let Some(mut c) = ui.server.take() {
                let _ = c.kill();
                let _ = c.wait();
            }
            unsafe {
                EnableWindow(ui.load, 1);
                EnableWindow(ui.unload, 0);
                EnableWindow(ui.send, 0);
            }
        });
        set_status("unloaded -- the memory is back");
    }

    fn notify() {
        UI.with(|u| {
            if let Some(ui) = u.borrow().as_ref() {
                unsafe {
                    PostMessageW(ui.main, WM_APP_TICK, 0, 0);
                }
            }
        });
    }

    fn send_prompt() {
        if busy().load(Ordering::SeqCst) {
            return;
        }
        let (prompt, port, history) = UI.with(|u| {
            let b = u.borrow();
            let Some(ui) = b.as_ref() else {
                return (String::new(), 0u16, Vec::new());
            };
            (control_text(ui.input), ui.port, ui.history.clone())
        });
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return;
        }

        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                ui.answer.clear();
                unsafe {
                    SetWindowTextW(ui.input, wide("").as_ptr());
                    EnableWindow(ui.send, 0);
                }
            }
        });
        append_out(&format!("\r\n> {}\r\n\r\n", prompt.replace('\n', "\r\n")));

        {
            let mut s = shared().lock().unwrap();
            s.pending.clear();
            s.finished = false;
            s.tokens = 0;
            s.started = Some(std::time::Instant::now());
            s.status = "thinking".into();
        }
        busy().store(true, Ordering::SeqCst);

        let p2 = prompt.clone();
        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                ui.history.push(("user".into(), p2));
            }
        });

        std::thread::spawn(move || {
            let hist: Vec<(String, String)> = history;
            client::chat(port, &hist, &prompt, 512, &mut |ev| match ev {
                client::Event::Token(t) => {
                    let mut s = shared().lock().unwrap();
                    s.pending.push_str(&t);
                    s.tokens += 1;
                    drop(s);
                    notify();
                }
                client::Event::Done => {
                    let mut s = shared().lock().unwrap();
                    s.finished = true;
                    s.status = "ready".into();
                    drop(s);
                    notify();
                }
                client::Event::Failed(m) => {
                    let mut s = shared().lock().unwrap();
                    s.pending.push_str(&format!("\n[{m}]\n"));
                    s.finished = true;
                    s.status = m;
                    drop(s);
                    notify();
                }
            });
        });
    }

    /// Drain what the worker produced. Runs on the UI thread only.
    fn drain() {
        let (text, finished, tokens, elapsed) = {
            let mut s = shared().lock().unwrap();
            let t = std::mem::take(&mut s.pending);
            let e = s.started.map(|i| i.elapsed().as_secs_f64()).unwrap_or(0.0);
            (t, s.finished, s.tokens, e)
        };
        if !text.is_empty() {
            // EDIT controls want CRLF; a bare LF renders as a box.
            append_out(&text.replace("\r\n", "\n").replace('\n', "\r\n"));
            UI.with(|u| {
                if let Some(ui) = u.borrow_mut().as_mut() {
                    ui.answer.push_str(&text);
                }
            });
        }
        if tokens > 0 && elapsed > 0.0 {
            let rate = tokens as f64 / elapsed;
            shared().lock().unwrap().status = format!("{tokens} tokens, {rate:.2} tok/s");
        }
        if finished && busy().swap(false, Ordering::SeqCst) {
            UI.with(|u| {
                if let Some(ui) = u.borrow_mut().as_mut() {
                    let a = std::mem::take(&mut ui.answer);
                    if !a.is_empty() {
                        ui.history.push(("assistant".into(), a));
                    }
                    unsafe {
                        EnableWindow(ui.send, 1);
                    }
                }
            });
        }
        if finished {
            UI.with(|u| {
                if let Some(ui) = u.borrow().as_ref() {
                    if ui.server.is_some() {
                        unsafe {
                            EnableWindow(ui.send, 1);
                        }
                    }
                }
            });
        }
        UI.with(|u| {
            if let Some(ui) = u.borrow().as_ref() {
                unsafe {
                    InvalidateRect(ui.main, std::ptr::null(), 0);
                }
            }
        });
    }

    unsafe fn layout(hwnd: HWND) {
        let mut r = RECT::default();
        GetClientRect(hwnd, &mut r);
        let w = r.right - r.left;
        let h = r.bottom - r.top;

        UI.with(|u| {
            let b = u.borrow();
            let Some(ui) = b.as_ref() else { return };
            let top = PAD * 2 + LOGO_PX + 26;
            let btn_h = 30;
            let list_h = h - top - PAD * 4 - btn_h * 2 - 46;

            MoveWindow(ui.list, PAD, top, SIDEBAR - PAD * 2, list_h.max(60), 1);
            let by = top + list_h.max(60) + PAD;
            let bw = (SIDEBAR - PAD * 3) / 2;
            MoveWindow(ui.load, PAD, by, bw, btn_h, 1);
            MoveWindow(ui.unload, PAD * 2 + bw, by, bw, btn_h, 1);
            MoveWindow(ui.refresh_handle(), PAD, by + btn_h + 8, bw, btn_h, 1);
            MoveWindow(ui.cache, PAD * 2 + bw, by + btn_h + 8, bw / 2 - 4, btn_h, 1);
            MoveWindow(
                ui.threads,
                PAD * 2 + bw + bw / 2 + 4,
                by + btn_h + 8,
                bw / 2 - 4,
                btn_h,
                1,
            );

            let rx = SIDEBAR;
            let rw = w - rx - PAD;
            let in_h = 92;
            let out_h = h - PAD * 3 - in_h - 34;
            MoveWindow(ui.out, rx, PAD, rw, out_h.max(80), 1);
            MoveWindow(ui.input, rx, PAD * 2 + out_h.max(80), rw - 110, in_h, 1);
            MoveWindow(
                ui.send,
                rx + rw - 100,
                PAD * 2 + out_h.max(80),
                100,
                in_h,
                1,
            );
        });
    }

    impl Ui {
        fn refresh_handle(&self) -> HWND {
            unsafe { GetDlgItem(self.main, ID_REFRESH) }
        }
    }

    unsafe fn paint(hwnd: HWND) {
        let mut ps: PAINTSTRUCT = std::mem::zeroed();
        let hdc = BeginPaint(hwnd, &mut ps);
        let mut r = RECT::default();
        GetClientRect(hwnd, &mut r);

        UI.with(|u| {
            let b = u.borrow();
            let Some(ui) = b.as_ref() else { return };
            FillRect(hdc, &r, ui.black);
            SetBkMode(hdc, TRANSPARENT);
            SetTextColor(hdc, WHITE);

            // The logo, thresholded to two values and blitted as a DIB. Drawn
            // rather than shipped as a second asset: it is the same array the
            // terminal banner prints, from the same SVG.
            let (lw, lh) = art::logo_size();
            let mono = art::logo_mono();
            let mut px = vec![0u8; lw * lh * 4];
            for y in 0..lh {
                for x in 0..lw {
                    // A DIB with a positive height is bottom-up, so the source
                    // row is mirrored here rather than the image being upside
                    // down on screen.
                    let v = if mono[(lh - 1 - y) * lw + x] {
                        255u8
                    } else {
                        0u8
                    };
                    let i = (y * lw + x) * 4;
                    px[i] = v;
                    px[i + 1] = v;
                    px[i + 2] = v;
                    px[i + 3] = 0;
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
                PAD + (SIDEBAR - PAD * 2 - LOGO_PX) / 2,
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

            SelectObject(hdc, ui.font as HGDIOBJ);
            let name = wide("C H A O S");
            TextOutW(hdc, PAD + 4, PAD + LOGO_PX + 6, name.as_ptr(), 9);

            // The rule between the sidebar and the conversation. A line, not a
            // panel, because a filled panel would need a third value.
            let pen = CreatePen(0, 1, WHITE);
            let old = SelectObject(hdc, pen);
            Rectangle(hdc, SIDEBAR - PAD, -2, SIDEBAR - PAD + 1, r.bottom + 2);
            SelectObject(hdc, old);
            DeleteObject(pen);

            SelectObject(hdc, ui.mono as HGDIOBJ);
            let s = shared().lock().unwrap();
            let status = if s.status.is_empty() {
                "idle".to_string()
            } else {
                s.status.clone()
            };
            drop(s);
            let line = wide(&status);
            TextOutW(
                hdc,
                SIDEBAR,
                r.bottom - 24,
                line.as_ptr(),
                status.encode_utf16().count() as i32,
            );
            let hint = "cache GiB / threads";
            let hw = wide(hint);
            TextOutW(hdc, PAD, r.bottom - 24, hw.as_ptr(), hint.len() as i32);
        });

        EndPaint(hwnd, &ps);
    }

    /// Paint one owner-drawn control in the two-value palette.
    ///
    /// Buttons and the list selection are the two places Windows insists on its
    /// own colours -- a themed push button ignores `WM_CTLCOLORBTN`, and the
    /// selection bar is the system highlight, which is blue. Both are drawn
    /// here instead. Emphasis is inversion, since there is no third value to
    /// reach for: selected and pressed are white-on-black turned inside out.
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

        // A one-pixel rule around a button, so it reads as a control without a
        // fill that would need a third value.
        // The frame is how a button says it can be pressed. Omitted when
        // disabled: with only two values, *removing* the rule is the quiet
        // signal and inverting the whole control is the loud one, and the loud
        // one was landing on precisely the controls that do nothing.
        if di.CtlType != ODT_LISTBOX && !disabled {
            let pen = CreatePen(0, 1, fg);
            let old_pen = SelectObject(di.hDC, pen);
            let hollow = CreateSolidBrush(bg);
            let old_brush = SelectObject(di.hDC, hollow as HGDIOBJ);
            Rectangle(
                di.hDC,
                di.rcItem.left,
                di.rcItem.top,
                di.rcItem.right,
                di.rcItem.bottom,
            );
            SelectObject(di.hDC, old_brush);
            SelectObject(di.hDC, old_pen);
            DeleteObject(hollow as HGDIOBJ);
            DeleteObject(pen as HGDIOBJ);
        }

        let text = if di.CtlType == ODT_LISTBOX {
            if di.itemID == u32::MAX {
                return;
            }
            let n = SendMessageW(di.hwndItem, LB_GETTEXTLEN, di.itemID as WPARAM, 0);
            if n <= 0 {
                return;
            }
            let mut buf = vec![0u16; n as usize + 1];
            SendMessageW(
                di.hwndItem,
                LB_GETTEXT,
                di.itemID as WPARAM,
                buf.as_mut_ptr() as LPARAM,
            );
            String::from_utf16_lossy(&buf[..n as usize])
        } else {
            let n = GetWindowTextLengthW(di.hwndItem);
            if n <= 0 {
                return;
            }
            let mut buf = vec![0u16; n as usize + 1];
            let got = GetWindowTextW(di.hwndItem, buf.as_mut_ptr(), n + 1);
            String::from_utf16_lossy(&buf[..got.max(0) as usize])
        };

        SetBkMode(di.hDC, TRANSPARENT);
        SetTextColor(di.hDC, fg);

        let w: Vec<u16> = text.encode_utf16().collect();
        let (x, y) = if di.CtlType == ODT_LISTBOX {
            (di.rcItem.left + 6, di.rcItem.top + 3)
        } else {
            // Centred by measurement would need GetTextExtentPoint32; the
            // buttons carry short fixed labels, so a fraction of the width is
            // close enough and cannot be wrong by more than a few pixels.
            let cw = (di.rcItem.right - di.rcItem.left) / 2;
            let approx = (text.len() as i32 * 7) / 2;
            (
                di.rcItem.left + (cw - approx).max(6),
                di.rcItem.top + (di.rcItem.bottom - di.rcItem.top) / 2 - 9,
            )
        };
        TextOutW(di.hDC, x, y, w.as_ptr(), w.len() as i32);
    }

    unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        match msg {
            WM_SIZE => {
                layout(hwnd);
                InvalidateRect(hwnd, std::ptr::null(), 1);
                0
            }
            WM_PAINT => {
                paint(hwnd);
                0
            }
            // Every control repainted to the two-value palette. Without these
            // the list and the boxes come up in the system's greys, which is
            // the whole design gone.
            WM_CTLCOLOREDIT | WM_CTLCOLORLISTBOX | WM_CTLCOLORSTATIC | WM_CTLCOLORBTN => {
                SetTextColor(wp as HDC, WHITE);
                SetBkColor(wp as HDC, BLACK);
                UI.with(|u| {
                    u.borrow()
                        .as_ref()
                        .map(|ui| ui.black as LRESULT)
                        .unwrap_or(0)
                })
            }
            WM_APP_TICK => {
                drain();
                0
            }
            WM_DRAWITEM => {
                let di = &*(lp as *const DRAWITEMSTRUCT);
                draw_item(di);
                1
            }
            WM_COMMAND => {
                let id = (wp & 0xFFFF) as i32;
                let code = ((wp >> 16) & 0xFFFF) as u16;
                match (id, code) {
                    (ID_LOAD, BN_CLICKED) => load_model(),
                    (ID_UNLOAD, BN_CLICKED) => unload_model(),
                    (ID_SEND, BN_CLICKED) => send_prompt(),
                    (ID_REFRESH, BN_CLICKED) => rescan(),
                    (ID_LIST, LBN_SELCHANGE) => {}
                    _ => {}
                }
                0
            }
            WM_CLOSE => {
                // Kill the child before the window goes: an orphaned server
                // holding 7 GiB is exactly the failure this project has a rule
                // about, and it would outlive the app that started it.
                unload_model();
                DestroyWindow(hwnd);
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
