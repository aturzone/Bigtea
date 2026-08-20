//! Chaos, as a window.
//!
//! A real Win32 application: `RegisterClassW`, a message loop, a menu bar,
//! native controls, GDI painting. No browser, no webview, no HTML, and no GUI
//! dependency -- the whole surface it uses is declared in `win32.rs` against
//! libraries Windows already ships.
//!
//! **Four pages, one of which owns the screen at a time.** The first version of
//! this window grew a control at a time into a single 380px column, and Atur's
//! verdict was exact: *"why is all click in one slot"*. `nav.rs` is the answer
//! written down as data -- where every control lives -- and `theme.rs` is the
//! palette, so nothing in this file names a colour. Both are testable without a
//! window server, which is most of what this file used to get wrong.
//!
//! The design follows Hermes' own `apps/desktop/DESIGN.md`, which Atur asked
//! this to match: flat rather than boxed, one primitive per concern, tokens
//! rather than literals, durable destinations, and expensive surfaces staying
//! mounted while hidden.
//!
//! The engine runs as a child `chaos-serve` process; `client.rs` says why at
//! length. The short version is that unloading a model has to actually free
//! 7 GiB, and that a second in-process construction path is where this codebase
//! has historically hidden its worst bug.

#![cfg_attr(not(windows), allow(dead_code))]
// The window subsystem, so double-clicking it does not also open a console.
// `main` still runs; only the console allocation is suppressed.
#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(not(windows))]
fn main() {
    eprintln!(
        "chaos-app is the Windows application; on this platform use chaos-run or chaos-serve."
    );
    std::process::exit(1);
}

#[cfg(windows)]
fn main() {
    windows_app::install_panic_hook();
    windows_app::run();
}

#[cfg(windows)]
mod windows_app {
    use chaos_app::choices::{self, Choice};
    use chaos_app::download::Download;
    use chaos_app::loading;
    use chaos_app::nav::{self, Page};
    use chaos_app::theme::{self, metric, size, weight, Mode, Rgb, Theme};
    use chaos_app::win32::*;
    use chaos_app::{art, catalog, client, models, settings, update};
    use std::cell::RefCell;
    use std::process::{Child, Command};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    /// The clock behind the strip and the monitor page. One second: uptime is
    /// shown to the second, and free memory moves slowly.
    const TICK_MS: u32 = 1000;
    const TIMER_ID: usize = 1;

    /// Below this the rail plus a page has nowhere to put anything, so the
    /// window refuses to get smaller rather than clipping its own controls --
    /// which is how the old sidebar came to show "gemma3-27b Q4_K_M 16.5 GB nee".
    const MIN_W: i32 = 940;
    const MIN_H: i32 = 620;

    const RELEASES_URL: &str = "https://github.com/aturzone/Chaos/releases";
    const MANUAL_URL: &str = "https://github.com/aturzone/Chaos/blob/main/docs/APP.md";

    /// State the worker thread and the window both touch.
    #[derive(Default)]
    struct Shared {
        /// Text produced since the UI last drained it.
        pending: String,
        /// Set when the answer is complete, so the UI can re-enable SEND.
        finished: bool,
        tokens: u32,
        started: Option<Instant>,
        status: String,
        /// Set when `chaos-pull` exits, so the UI stops watching the files.
        download_done: bool,
        /// A message box the UI thread must put up: its text, and whether it is
        /// good news.
        ///
        /// **A worker cannot show one itself.** `MessageBoxW` with an owner
        /// window belonging to another thread is undefined, and what it does
        /// here is nothing at all -- the connection test ran, passed, set the
        /// status line, and displayed no dialog whatsoever.
        report: Option<(String, bool)>,
        /// What `chaos-serve` wrote to stderr, kept so a server that dies before
        /// it is ready can say why.
        ///
        /// **It used to go nowhere.** The child is spawned with
        /// `CREATE_NO_WINDOW` and inherits no console, so its one-line reason
        /// for exiting -- "Only one usage of each socket address ... (os error
        /// 10048)" -- was written to a handle nobody held. The window showed
        /// "the model did not come up" after ten minutes of polling and the
        /// actual sentence was lost.
        serve_err: String,
        /// What the last update check found, kept so "Install update" knows
        /// which file to fetch without asking GitHub a second time.
        update: Option<update::Outcome>,
        /// Set when a check has produced something the window has not shown
        /// yet. Separate from `update` because the outcome outlives its
        /// announcement -- the menu item stays usable after the notice is gone.
        update_news: bool,
        /// Whether the announcement should be a dialog or only the status line.
        /// The check at startup is quiet unless there is news.
        update_asked: bool,
        /// Set when the installer is running, so the window closes and stops
        /// holding `chaos-app.exe` open. Windows will not let the installer
        /// overwrite a binary that is still executing.
        update_quit: bool,
    }

    /// Which set of models the MODELS page lists.
    #[derive(PartialEq, Clone, Copy)]
    enum Tab {
        Installed,
        Available,
    }

    /// The faces, made once. Three sizes and a monospace, per `theme::size`.
    struct Fonts {
        display: HFONT,
        heading: HFONT,
        body: HFONT,
        body_bold: HFONT,
        small: HFONT,
        mono: HFONT,
        mark: HFONT,
    }

    /// The solid brushes the palette needs. Remade when the theme changes,
    /// because `WM_CTLCOLOR*` has to answer with a brush, not a colour.
    struct Brushes {
        bg: HBRUSH,
    }

    /// Everything the window knows.
    ///
    /// **No window handles live here.** They are fetched by id through `ctl()`,
    /// which reads the main handle out of an atomic and asks Windows -- so no
    /// caller has to hold a borrow open in order to find a control. That was
    /// the shape of the bug that made every click fatal.
    struct Ui {
        theme: Theme,
        page: Page,
        tab: Tab,
        fonts: Fonts,
        brushes: Brushes,
        entries: Vec<models::Entry>,
        /// How many files each installed entry is spread across, counted once
        /// per rescan. **Not counted while painting**: that is a directory scan,
        /// and the transcript repaints on every token.
        entry_files: Vec<usize>,
        offers: Vec<catalog::Offer>,
        free_bytes: u64,
        total_bytes: u64,
        server: Option<Child>,
        port: u16,
        /// The label of the model currently served, if any.
        loaded: Option<String>,
        /// When it came up, for the uptime on the strip.
        loaded_at: Option<Instant>,
        /// Tokens this server has produced since it started.
        served: u64,
        /// The last measured generation rate, kept so the strip has something
        /// to show between answers.
        last_rate: f64,
        history: Vec<(String, String)>,
        answer: String,
        cfg: settings::Settings,
        /// What this machine is, so the settings page can offer values that
        /// make sense on it rather than an empty box.
        machine: choices::Machine,
        /// A download in flight, watched by the bytes it puts on disk.
        download: Option<Download>,
        /// The options each dropdown currently holds, by control id. Kept so
        /// the note under a box follows the *selection* rather than repeating a
        /// static sentence, and so saving reads the value rather than the label.
        lists: std::collections::HashMap<i32, Vec<Choice>>,
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

    /// The main window handle, readable from any thread.
    ///
    /// `UI` is a `thread_local!`, so a worker thread sees `None` in it -- its
    /// own copy was never initialised. Every background task here needs the
    /// window handle to wake the UI, so it lives in an atomic as well.
    fn main_window() -> &'static std::sync::atomic::AtomicUsize {
        static H: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        &H
    }

    fn main_hwnd() -> HWND {
        main_window().load(Ordering::SeqCst) as HWND
    }

    /// A settings control's handle, for the paint path. Named separately so the
    /// call inside `paint_settings` reads as what it is: a handle lookup, which
    /// `GetDlgItem` performs without dispatching a message.
    fn sel_of(id: i32) -> HWND {
        ctl(id)
    }

    /// A control by id, without borrowing anything.
    ///
    /// The single most useful line in this file: because handles are found this
    /// way rather than read out of `UI`, no action function needs a borrow open
    /// while it talks to Windows.
    fn ctl(id: i32) -> HWND {
        let h = main_hwnd();
        if h.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { GetDlgItem(h, id) }
    }

    /// Wake the UI thread so it drains what a worker produced.
    ///
    /// **This used to read `UI`, from the worker thread**, where the
    /// thread-local is `None` -- so it posted nothing, `drain` never ran, and
    /// every generated token was received and silently discarded.
    fn notify() {
        let h = main_hwnd();
        if !h.is_null() {
            unsafe {
                PostMessageW(h, WM_APP_TICK, 0, 0);
            }
        }
    }

    fn repaint() {
        let h = main_hwnd();
        if !h.is_null() {
            unsafe {
                InvalidateRect(h, std::ptr::null(), 0);
            }
        }
    }

    /// Make a crash say something.
    ///
    /// The release profile sets `panic = "abort"`, so a panic is not an unwind
    /// that can be caught -- the process simply vanishes. With no console
    /// attached that means no message, no log and nothing to report. The hook
    /// runs *before* the abort, so it still gets to write the file and put a
    /// box on screen naming it.
    pub fn install_panic_hook() {
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
            let text = format!("Chaos crashed.\n\n{what}\nat {where_}\n");
            let path = std::env::temp_dir().join("chaos-app-crash.log");
            let _ = std::fs::write(&path, &text);
            unsafe {
                let msg = format!("{text}\nWritten to:\n{}", path.display());
                MessageBoxW(
                    std::ptr::null_mut(),
                    wide(&msg).as_ptr(),
                    wide("Chaos").as_ptr(),
                    MB_OK | MB_ICONERROR,
                );
            }
        }));
    }

    pub fn run() {
        // **Before any window exists**, because awareness cannot be changed
        // afterwards. Without it Windows renders the app at 96 DPI and stretches
        // the result: soft text, and every coordinate in a made-up space.
        chaos_app::win32::become_dpi_aware();
        let cfg = settings::Settings::load();

        unsafe {
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("ChaosAppWindow");

            let wc = WNDCLASSW {
                style: 0,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinst,
                hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW as *const u16),
                // Null: `WM_ERASEBKGND` is answered here instead, so Windows
                // never paints a ground we are about to paint over. That flash
                // of the wrong colour on every resize is the whole of it.
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
            };
            if RegisterClassW(&wc) == 0 {
                return;
            }

            let title = wide("Chaos");
            let geom = opening_geometry();
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                // `WS_CLIPCHILDREN`: the timer repaints this window every
                // second for the uptime and the download bar, and without it
                // every child control repaints too -- a visible flash across
                // the whole window, once a second, forever.
                WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
                geom.0,
                geom.1,
                geom.2,
                geom.3,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinst,
                std::ptr::null_mut(),
            );
            if hwnd.is_null() {
                return;
            }

            set_window_icon(hwnd, hinst);
            main_window().store(hwnd as usize, Ordering::SeqCst);

            build_controls(hwnd, hinst, cfg);
            build_menu(hwnd);
            sync_titlebar();
            fill_settings_page();
            show_page(Page::Chat);
            SetTimer(hwnd, TIMER_ID, TICK_MS, 0);

            ShowWindow(hwnd, SW_SHOW);
            UpdateWindow(hwnd);

            // Ask once, on a worker, after the window is up. Quiet unless
            // there is something newer: this is how a user finds out a release
            // exists without being told to go and look, and it costs one
            // request against a static JSON file. `CHAOS_NO_UPDATE_CHECK`
            // turns it off for anyone who would rather it did not.
            if std::env::var_os("CHAOS_NO_UPDATE_CHECK").is_none() {
                check_for_updates(false);
            }

            let accel = accelerators();
            let mut msg = MSG::default();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                // Accelerators first, or `Ctrl+2` is swallowed by whichever
                // edit control happens to have focus.
                if !accel.is_null() && TranslateAcceleratorW(hwnd, accel, &mut msg) != 0 {
                    continue;
                }
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    // -- the menu ------------------------------------------------------------

    /// The menu bar.
    ///
    /// **Every command in the app is in here**, which is the point: Atur asked
    /// *"where is the menu, where are the options"*, and a Windows application
    /// with no menu bar has answered that with "nowhere". The rail and the
    /// buttons are shortcuts to these, never the only route.
    unsafe fn build_menu(hwnd: HWND) {
        let bar = CreateMenu();

        let file = CreatePopupMenu();
        add(file, nav::IDM_RESCAN, "&Rescan models\tF5");
        add(file, nav::IDM_OPEN_MODELS_DIR, "Open models &folder");
        sep(file);
        add(file, nav::IDM_EXIT, "E&xit");
        popup(bar, file, "&File");

        let model = CreatePopupMenu();
        add(model, nav::IDM_LOAD, "&Load selected\tCtrl+L");
        add(model, nav::IDM_STOP, "&Stop running model");
        sep(model);
        add(model, nav::IDM_DOWNLOAD, "&Download selected");
        add(model, nav::IDM_DELETE, "De&lete selected");
        sep(model);
        add(model, nav::IDM_COPY_ENDPOINT, "&Copy endpoint\tCtrl+E");
        add(model, nav::IDM_API_KEY, "Require an &API key");
        add(model, nav::IDM_TEST_CONNECTION, "&Test the connection");
        popup(bar, model, "&Model");

        let view = CreatePopupMenu();
        for p in nav::PAGES {
            let id = menu_of_page(p);
            add(
                view,
                id,
                &format!("{}\tCtrl+{}", p.title(), p.accel() as char),
            );
        }
        sep(view);
        add(view, nav::IDM_THEME_LIGHT, "&Light");
        add(view, nav::IDM_THEME_DARK, "&Dark");
        popup(bar, view, "&View");

        let help = CreatePopupMenu();
        add(help, nav::IDM_MANUAL, "&Manual");
        add(help, nav::IDM_CHECK_UPDATE, "Check for &updates");
        add(help, nav::IDM_INSTALL_UPDATE, "&Install update...");
        sep(help);
        add(help, nav::IDM_RELEASES, "&Releases");
        add(help, nav::IDM_CRASH_LOG, "Open &crash log");
        sep(help);
        add(help, nav::IDM_ABOUT, "&About Chaos");
        popup(bar, help, "&Help");

        SetMenu(hwnd, bar);
        DrawMenuBar(hwnd);
    }

    fn menu_of_page(p: Page) -> i32 {
        match p {
            Page::Chat => nav::IDM_PAGE_CHAT,
            Page::Models => nav::IDM_PAGE_MODELS,
            Page::Monitor => nav::IDM_PAGE_MONITOR,
            Page::Settings => nav::IDM_PAGE_SETTINGS,
        }
    }

    unsafe fn add(menu: HMENU, id: i32, text: &str) {
        AppendMenuW(menu, MF_STRING, id as usize, wide(text).as_ptr());
    }

    unsafe fn sep(menu: HMENU) {
        AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null());
    }

    unsafe fn popup(bar: HMENU, menu: HMENU, text: &str) {
        AppendMenuW(bar, MF_POPUP, menu as usize, wide(text).as_ptr());
    }

    /// Keyboard routes to the same commands the menu names.
    ///
    /// Hermes' rule, and a good one: *"a command may have keyboard, palette,
    /// and visible affordances, but they invoke the same action"*. These post
    /// the same `WM_COMMAND` a menu pick does.
    unsafe fn accelerators() -> HACCEL {
        let mut a: Vec<ACCEL> = nav::PAGES
            .iter()
            .map(|&p| ACCEL {
                fVirt: FVIRTKEY | FCONTROL,
                key: p.accel() as u16,
                cmd: menu_of_page(p) as u16,
            })
            .collect();
        a.push(ACCEL {
            fVirt: FVIRTKEY,
            key: VK_F5,
            cmd: nav::IDM_RESCAN as u16,
        });
        a.push(ACCEL {
            fVirt: FVIRTKEY | FCONTROL,
            key: b'L' as u16,
            cmd: nav::IDM_LOAD as u16,
        });
        a.push(ACCEL {
            fVirt: FVIRTKEY | FCONTROL,
            key: b'E' as u16,
            cmd: nav::IDM_COPY_ENDPOINT as u16,
        });
        // Ctrl+Enter sends, from anywhere -- including the composer, which is
        // multi-line precisely so plain Enter can make a paragraph.
        a.push(ACCEL {
            fVirt: FVIRTKEY | FCONTROL,
            key: VK_RETURN,
            cmd: nav::ID_SEND as u16,
        });
        CreateAcceleratorTableW(a.as_ptr(), a.len() as i32)
    }

    /// Grey out what cannot be done, each time a menu opens.
    ///
    /// A menu that offers "Stop running model" with nothing running is a menu
    /// that has to be tried in order to be understood.
    unsafe fn sync_menu(hwnd: HWND) {
        let menu = GetMenu(hwnd);
        if menu.is_null() {
            return;
        }
        let (page, tab, running, mode) = UI.with(|u| {
            u.borrow()
                .as_ref()
                .map(|ui| (ui.page, ui.tab, ui.loaded.is_some(), ui.theme.mode))
                .unwrap_or((Page::Chat, Tab::Installed, false, Mode::Light))
        });
        let installed = tab == Tab::Installed;
        let on_models = page == Page::Models;
        let enable = |id: i32, yes: bool| {
            EnableMenuItem(
                menu,
                id as u32,
                MF_BYCOMMAND | if yes { MF_ENABLED } else { MF_GRAYED },
            );
        };
        enable(nav::IDM_LOAD, on_models && installed && !running);
        enable(nav::IDM_STOP, running);
        enable(nav::IDM_DOWNLOAD, on_models && !installed);
        enable(nav::IDM_DELETE, on_models && installed && !running);
        enable(nav::IDM_COPY_ENDPOINT, running);
        enable(nav::IDM_TEST_CONNECTION, running);
        // Ticked when a key is required, so the menu answers "is one set?"
        // without anybody having to open the settings file.
        let keyed = UI.with(|u| {
            u.borrow()
                .as_ref()
                .map(|ui| ui.cfg.api_key.is_some())
                .unwrap_or(false)
        });
        CheckMenuItem(
            menu,
            nav::IDM_API_KEY as u32,
            MF_BYCOMMAND | if keyed { MF_CHECKED } else { MF_UNCHECKED },
        );

        CheckMenuRadioItem(
            menu,
            nav::IDM_PAGE_CHAT as u32,
            nav::IDM_PAGE_SETTINGS as u32,
            menu_of_page(page) as u32,
            MF_BYCOMMAND,
        );
        let theme_id = match mode {
            Mode::Light => nav::IDM_THEME_LIGHT,
            Mode::Dark => nav::IDM_THEME_DARK,
        };
        CheckMenuRadioItem(
            menu,
            nav::IDM_THEME_LIGHT as u32,
            nav::IDM_THEME_DARK as u32,
            theme_id as u32,
            MF_BYCOMMAND,
        );
    }

    // -- construction --------------------------------------------------------

    unsafe fn make_font(px: i32, weight: i32, face: &str) -> HFONT {
        let name = wide(face);
        // `iQuality = 5` is CLEARTYPE_QUALITY: without it small text on a light
        // ground is noticeably rougher than every other window on the desktop.
        CreateFontW(px, 0, 0, 0, weight, 0, 0, 0, 1, 0, 0, 5, 0, name.as_ptr())
    }

    /// Where the window opens: `(x, y, width, height)`.
    ///
    /// # Why this is not four constants
    ///
    /// It was. `(120, 80, 1180, 780)` puts the bottom edge at 860 on a desktop
    /// whose work area is 816 tall -- the machine this was written on has a
    /// scaled 1536x864 primary screen -- so the window opened with its lower
    /// strip under the taskbar, every time, on first run. A fixed size cannot
    /// be right on an unknown screen.
    ///
    /// So: the preferred size, shrunk to fit the work area with a margin, and
    /// centred in it. Centred rather than at a corner because the work area's
    /// origin is not always `(0, 0)` -- a taskbar on the left or top moves it,
    /// and a second monitor to the left makes it negative.
    fn opening_geometry() -> (i32, i32, i32, i32) {
        const WANT_W: i32 = 1180;
        const WANT_H: i32 = 780;
        const MARGIN: i32 = 40;
        let Some((l, t, r, b)) = chaos_app::win32::work_area() else {
            return (120, 80, WANT_W, WANT_H);
        };
        let (aw, ah) = ((r - l).max(320), (b - t).max(240));
        let w = WANT_W.min(aw - MARGIN);
        let h = WANT_H.min(ah - MARGIN);
        (l + (aw - w) / 2, t + (ah - h) / 2, w, h)
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
        // Created hidden, with no `WS_VISIBLE`. `show_page` is the only thing
        // that reveals a control, so one missing from `nav::controls` never
        // appears at all -- a visible bug rather than a silent overlap.
        CreateWindowExW(
            0,
            c.as_ptr(),
            t.as_ptr(),
            WS_CHILD | style,
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

    unsafe fn button(parent: HWND, text: &str, id: i32, hinst: HINSTANCE) -> HWND {
        child(parent, "BUTTON", text, BS_OWNERDRAW | WS_TABSTOP, id, hinst)
    }

    unsafe fn build_controls(hwnd: HWND, hinst: HINSTANCE, cfg: settings::Settings) {
        let fonts = Fonts {
            display: make_font(size::DISPLAY, weight::BOLD, theme::FACE_UI),
            heading: make_font(size::HEADING, weight::MEDIUM, theme::FACE_UI),
            body: make_font(size::BODY, weight::REGULAR, theme::FACE_UI),
            body_bold: make_font(size::BODY, weight::MEDIUM, theme::FACE_UI),
            small: make_font(size::SMALL, weight::REGULAR, theme::FACE_UI),
            mono: make_font(size::MONO, weight::REGULAR, theme::FACE_MONO),
            mark: make_font(size::MARK, weight::BOLD, theme::FACE_UI),
        };

        // The shell.
        for p in nav::PAGES {
            button(hwnd, p.label(), nav::nav_id(p), hinst);
        }
        button(hwnd, "STOP", nav::ID_STRIP_STOP, hinst);

        // Chat.
        child(
            hwnd,
            "EDIT",
            "",
            ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL | WS_BORDER,
            nav::ID_OUT,
            hinst,
        );
        child(
            hwnd,
            "EDIT",
            "",
            ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN | WS_VSCROLL | WS_BORDER | WS_TABSTOP,
            nav::ID_IN,
            hinst,
        );
        button(hwnd, "SEND", nav::ID_SEND, hinst);
        button(hwnd, "CLEAR", nav::ID_CLEAR, hinst);

        // Models.
        button(hwnd, "INSTALLED", nav::ID_TAB_INSTALLED, hinst);
        button(hwnd, "AVAILABLE", nav::ID_TAB_AVAILABLE, hinst);
        child(
            hwnd,
            "LISTBOX",
            "",
            LBS_NOTIFY | LBS_OWNERDRAWFIXED | LBS_HASSTRINGS | WS_VSCROLL | WS_BORDER,
            nav::ID_LIST,
            hinst,
        );
        button(hwnd, "LOAD", nav::ID_LOAD, hinst);
        button(hwnd, "STOP", nav::ID_UNLOAD, hinst);
        button(hwnd, "DOWNLOAD", nav::ID_GET, hinst);
        button(hwnd, "DELETE", nav::ID_DELETE, hinst);
        button(hwnd, "RESCAN", nav::ID_REFRESH, hinst);
        button(hwnd, "COPY ENDPOINT", nav::ID_COPY_ENDPOINT, hinst);

        // Settings: every field the file holds, which is the point of the page.
        //
        // **A field with a sensible short list is a dropdown, not a box.** An
        // empty text box asks "how many threads?", which most people cannot
        // answer and which has a different right answer on every machine.
        // `choices::for_field` decides which is which, so the window does not
        // carry a second copy of that judgement.
        let probe = probe_machine();
        for f in nav::FIELDS {
            if choices::for_field(f.id, probe).is_some() {
                child(
                    hwnd,
                    "COMBOBOX",
                    "",
                    CBS_DROPDOWNLIST | CBS_OWNERDRAWFIXED | CBS_HASSTRINGS | WS_VSCROLL,
                    f.id,
                    hinst,
                );
            } else {
                child(hwnd, "EDIT", "", WS_BORDER | WS_TABSTOP, f.id, hinst);
            }
        }
        for f in nav::TOGGLES {
            button(hwnd, f.label, f.id, hinst);
        }
        button(hwnd, "SAVE", nav::ID_SAVE, hinst);
        button(hwnd, "RESET", nav::ID_RESET, hinst);
        button(hwnd, "BROWSE...", nav::ID_BROWSE_MODELS, hinst);

        // The transcript, the composer and the list carry measurements, so they
        // are monospaced; everything else is the UI face.
        let mono_controls = [nav::ID_OUT, nav::ID_IN, nav::ID_LIST];
        for p in nav::PAGES {
            for &id in nav::controls(p) {
                let f = if mono_controls.contains(&id) {
                    fonts.mono
                } else {
                    fonts.body
                };
                SendMessageW(GetDlgItem(hwnd, id), WM_SETFONT, f as WPARAM, 1);
            }
        }
        for id in nav::SHELL_CONTROLS {
            SendMessageW(GetDlgItem(hwnd, id), WM_SETFONT, fonts.body as WPARAM, 1);
        }

        // An owner-draw list uses a fixed row height that defaults to roughly
        // the system font's, which clipped the model name in half.
        SendMessageW(GetDlgItem(hwnd, nav::ID_LIST), LB_SETITEMHEIGHT, 0, 28);

        // An EDIT puts its text flush against the border otherwise, which on a
        // design built out of whitespace is the one control that has none.
        for id in [nav::ID_OUT, nav::ID_IN] {
            SendMessageW(
                GetDlgItem(hwnd, id),
                EM_SETMARGINS,
                EC_LEFTMARGIN | EC_RIGHTMARGIN,
                (10 | (10 << 16)) as LPARAM,
            );
        }
        for f in nav::FIELDS {
            let h = GetDlgItem(hwnd, f.id);
            if choices::for_field(f.id, probe).is_some() {
                // Every row of the list, then -- `usize::MAX` is `-1` -- the
                // closed box itself. The closed height is what Windows keeps
                // when it shrinks the control; `layout` sizes the rest.
                SendMessageW(h, CB_SETITEMHEIGHT, 0, metric::COMBO_ROW as LPARAM);
                SendMessageW(
                    h,
                    CB_SETITEMHEIGHT,
                    usize::MAX,
                    (metric::CONTROL - 6) as LPARAM,
                );
            } else {
                SendMessageW(
                    h,
                    EM_SETMARGINS,
                    EC_LEFTMARGIN | EC_RIGHTMARGIN,
                    (8 | (8 << 16)) as LPARAM,
                );
            }
        }

        let t = theme::theme(cfg.mode);
        let port = cfg.port;
        let machine = probe_machine();
        UI.with(|u| {
            *u.borrow_mut() = Some(Ui {
                theme: t,
                page: Page::Chat,
                tab: Tab::Installed,
                fonts,
                brushes: Brushes {
                    bg: CreateSolidBrush(t.bg),
                },
                entries: Vec::new(),
                entry_files: Vec::new(),
                offers: catalog::offers(),
                free_bytes: free_memory_bytes(),
                total_bytes: total_memory_bytes(),
                server: None,
                port,
                loaded: None,
                loaded_at: None,
                served: 0,
                last_rate: 0.0,
                history: Vec::new(),
                answer: String::new(),
                cfg,
                machine,
                download: None,
                lists: std::collections::HashMap::new(),
            })
        });
        rescan();
    }

    /// What the machine is, for the settings page.
    ///
    /// `chaos-probe` already answers this properly and has no dependencies of
    /// its own, so the app asks it rather than growing a second, worse copy.
    /// `measure_bandwidth: false` skips the only slow step -- the disk timing --
    /// because nothing on this page needs it and the window must not stall on
    /// startup.
    fn probe_machine() -> choices::Machine {
        let m = chaos_probe::Machine::probe(models::default_dir(), false);
        choices::Machine {
            cores: m.cpu_threads.max(1) as u32,
            total_ram: m.ram_total_bytes.unwrap_or(0),
            free_ram: m.ram_available_bytes.unwrap_or(0),
            gpu: !m.gpus.is_empty(),
        }
    }

    /// Match the title bar to the palette.
    ///
    /// The compositor draws it, not us, so it stays light however the client
    /// area is painted. A no-op on Windows builds too old to know the
    /// attribute, which is why the return value is not checked.
    fn sync_titlebar() {
        let dark = UI.with(|u| {
            u.borrow()
                .as_ref()
                .map(|ui| ui.theme.mode == Mode::Dark)
                .unwrap_or(false)
        });
        let h = main_hwnd();
        if h.is_null() {
            return;
        }
        let on: i32 = i32::from(dark);
        unsafe {
            DwmSetWindowAttribute(
                h,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                &on as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u32,
            );
        }
        // The scrollbars are the system's too, and they came up `#F0F0F0` on a
        // `#0D0D0E` page until this was here. Naming the control's theme class
        // is the documented way to move them; measured at `#171717` after.
        for id in [nav::ID_OUT, nav::ID_IN, nav::ID_LIST] {
            unsafe {
                set_control_theme(ctl(id), dark);
            }
        }
        // **The menu bar stays light in dark mode, and there is no fix here.**
        // It is non-client, so it does not follow the title-bar attribute.
        // `SetPreferredAppMode` -- uxtheme ordinal 135, what every dark Win32
        // app calls -- was tried both before window creation and after, with
        // `FlushMenuThemes`: the ordinals resolve on this build (10.0.26200)
        // and the bar still measures `#FFFFFF`. Owner-drawing the whole menu is
        // the only remaining route, and it needs the undocumented
        // `WM_UAHDRAWMENU` to do the bar's background. Not worth it for a strip
        // that is light on a theme most people will not switch to; `docs/APP.md`
        // says so plainly rather than leaving it to be noticed.
    }

    // -- pages ---------------------------------------------------------------

    /// Reveal one page and hide the rest.
    ///
    /// Controls are hidden, never destroyed. Hermes states the reason better
    /// than a comment here could: *"expensive stateful surfaces stay mounted
    /// when hidden -- visibility is not lifecycle"*. A transcript rebuilt every
    /// time you glance at the settings is a transcript you have lost.
    fn show_page(p: Page) {
        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                ui.page = p;
            }
        });

        // Borrow is closed. `ShowWindow` repaints, which asks the parent for
        // colours, which borrows -- doing this inside would abort the process.
        for q in nav::PAGES {
            let show = q == p;
            for &id in nav::controls(q) {
                unsafe {
                    ShowWindow(ctl(id), if show { SW_SHOW } else { SW_HIDE });
                }
            }
        }
        for id in nav::SHELL_CONTROLS {
            unsafe {
                ShowWindow(ctl(id), SW_SHOW);
            }
        }

        let h = main_hwnd();
        if !h.is_null() {
            unsafe {
                layout(h);
                InvalidateRect(h, std::ptr::null(), 1);
            }
        }
        // **Invalidating the parent does not repaint its children.** The rail
        // buttons are owner-drawn, so they only redraw when Windows sends them
        // a `WM_DRAWITEM`, and it only does that when the *button* is
        // invalidated. Without this the highlight followed the mouse rather
        // than the page: clicking SETTINGS in the rail worked because the click
        // repainted that one button, but View > Settings and Ctrl+4 left CHAT
        // lit while the settings page was on screen.
        //
        // The primary action moves with the page too (`weight_of`), so those
        // buttons are refreshed here as well rather than staying heavy on a
        // page they no longer belong to.
        for id in nav::SHELL_CONTROLS {
            let c = ctl(id);
            if !c.is_null() {
                unsafe {
                    InvalidateRect(c, std::ptr::null(), 1);
                }
            }
        }
        for &id in nav::controls(p) {
            let c = ctl(id);
            if !c.is_null() {
                unsafe {
                    InvalidateRect(c, std::ptr::null(), 1);
                }
            }
        }
        sync_enabled();
        // The page you arrive at gets the caret, so typing works without a
        // click first. Chat is the home surface, so there it is the composer.
        let focus = match p {
            Page::Chat => Some(nav::ID_IN),
            Page::Models => Some(nav::ID_LIST),
            Page::Settings => Some(nav::FIELDS[0].id),
            Page::Monitor => None,
        };
        if let Some(id) = focus {
            unsafe {
                SetFocus(ctl(id));
            }
        }
    }

    fn set_mode(m: Mode) {
        let old = UI.with(|u| {
            let mut b = u.borrow_mut();
            let ui = b.as_mut()?;
            if ui.theme.mode == m {
                return None;
            }
            let old = ui.brushes.bg;
            ui.theme = theme::theme(m);
            ui.brushes = Brushes {
                bg: unsafe { CreateSolidBrush(ui.theme.bg) },
            };
            ui.cfg.mode = m;
            let _ = ui.cfg.save();
            Some(old)
        });
        let Some(bg) = old else { return };
        // The replacement is already in place, so the old one is safe to
        // release. Leaking a brush per toggle would be invisible and permanent.
        unsafe {
            DeleteObject(bg as HGDIOBJ);
        }
        sync_titlebar();
        let h = main_hwnd();
        if !h.is_null() {
            unsafe {
                InvalidateRect(h, std::ptr::null(), 1);
            }
        }
    }

    /// Which controls are live, given what is running and which tab is showing.
    ///
    /// One function, called after anything that could change the answer, rather
    /// than an `EnableWindow` scattered through every action -- which is how
    /// the old window ended up with a SEND button that stayed grey after a
    /// model came up.
    fn sync_enabled() {
        // The selection is read first: `LB_GETCURSEL` is a `SendMessageW`, and
        // this file's rule is that Windows is never called with `UI` open.
        let sel = selection();
        let (installed, running, busy_now, unfinished) = UI.with(|u| {
            u.borrow()
                .as_ref()
                .map(|ui| {
                    (
                        ui.tab == Tab::Installed,
                        ui.loaded.is_some(),
                        busy().load(Ordering::SeqCst),
                        sel.and_then(|s| ui.entries.get(s))
                            .is_some_and(|e| e.incomplete.is_some()),
                    )
                })
                .unwrap_or((true, false, false, false))
        });
        let set = |id: i32, on: bool| unsafe {
            EnableWindow(ctl(id), i32::from(on));
        };
        // **An unfinished model cannot be loaded and can be finished.** Leaving
        // DOWNLOAD grey on the INSTALLED tab meant the one action that fixes
        // the problem was the one action not offered.
        set(nav::ID_LOAD, installed && !running && !unfinished);
        set(nav::ID_UNLOAD, running);
        set(nav::ID_GET, !installed || (unfinished && !running));
        set(nav::ID_DELETE, installed && !running);
        set(nav::ID_COPY_ENDPOINT, running);
        set(nav::ID_SEND, running && !busy_now);
        set(nav::ID_STRIP_STOP, running);
    }

    // -- machine facts -------------------------------------------------------

    #[repr(C)]
    struct MemStatus {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page: u64,
        avail_page: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemStatus) -> i32;
    }

    fn mem_status() -> Option<MemStatus> {
        unsafe {
            let mut m: MemStatus = std::mem::zeroed();
            m.length = std::mem::size_of::<MemStatus>() as u32;
            (GlobalMemoryStatusEx(&mut m) != 0).then_some(m)
        }
    }

    /// Physical memory currently free, for the "would this run here" verdict.
    ///
    /// `chaos-probe` reports this properly; the app needs one number and not a
    /// dependency on the probe crate, so it asks Windows directly.
    fn free_memory_bytes() -> u64 {
        mem_status().map(|m| m.avail_phys).unwrap_or(0)
    }

    fn total_memory_bytes() -> u64 {
        mem_status().map(|m| m.total_phys).unwrap_or(0)
    }

    // -- actions -------------------------------------------------------------

    fn rescan() {
        // Phase 1 -- read state and build the strings. Borrow ends here.
        let rows: Vec<String> = {
            let free = free_memory_bytes();
            let total = total_memory_bytes();
            UI.with(|u| {
                let mut b = u.borrow_mut();
                let Some(ui) = b.as_mut() else {
                    return Vec::new();
                };
                ui.free_bytes = free;
                ui.total_bytes = total;
                match ui.tab {
                    Tab::Installed => {
                        ui.entries = models::list();
                        // Counted here, once, rather than while painting: this
                        // is a directory scan per model and the window
                        // repaints on every generated token.
                        ui.entry_files = ui
                            .entries
                            .iter()
                            .map(|e| shards_of(&e.path).len())
                            .collect();
                        if ui.entries.is_empty() {
                            vec!["nothing installed -- open AVAILABLE to download one".to_string()]
                        } else {
                            ui.entries.iter().map(models::row).collect()
                        }
                    }
                    Tab::Available => ui.offers.iter().map(|o| catalog::row(o, free)).collect(),
                }
            })
        };

        // Phase 2 -- nothing borrowed. Windows may re-enter freely.
        let list = ctl(nav::ID_LIST);
        if list.is_null() {
            return;
        }
        unsafe {
            SendMessageW(list, LB_RESETCONTENT, 0, 0);
            for r in &rows {
                let t = wide(r);
                SendMessageW(list, LB_ADDSTRING, 0, t.as_ptr() as LPARAM);
            }
            SendMessageW(list, LB_SETCURSEL, 0, 0);
        }
        sync_enabled();
        repaint();
    }

    fn set_status(text: &str) {
        shared().lock().unwrap().status = text.to_string();
        repaint();
    }

    fn append_out(text: &str) {
        let out = ctl(nav::ID_OUT);
        if out.is_null() {
            return;
        }
        let w = wide(text);
        unsafe {
            // **The box is ES_READONLY, and a read-only EDIT ignores
            // EM_REPLACESEL entirely** -- no error, no text, nothing. Every
            // token the model produced was dropped here while the status line
            // cheerfully said "ready". Drop the flag for the append and put it
            // straight back, so the transcript still cannot be typed into.
            SendMessageW(out, EM_SETREADONLY, 0, 0);
            // Move the caret to the end, then replace an empty selection: the
            // only way to append to an EDIT without re-sending the whole
            // buffer, which for a long answer is quadratic.
            let len = GetWindowTextLengthW(out);
            SendMessageW(out, EM_SETSEL, len as WPARAM, len as LPARAM);
            SendMessageW(out, EM_REPLACESEL, 0, w.as_ptr() as LPARAM);
            SendMessageW(out, EM_SCROLLCARET, 0, 0);
            SendMessageW(out, EM_SETREADONLY, 1, 0);
        }
    }

    fn control_text(h: HWND) -> String {
        if h.is_null() {
            return String::new();
        }
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

    /// The row the list has selected.
    fn selection() -> Option<usize> {
        let list = ctl(nav::ID_LIST);
        if list.is_null() {
            return None;
        }
        let sel = unsafe { SendMessageW(list, LB_GETCURSEL, 0, 0) };
        (sel >= 0).then_some(sel as usize)
    }

    /// Start `chaos-serve` on the selected model, hidden.
    fn load_model() {
        let Some(sel) = selection() else {
            set_status("select a model first");
            return;
        };
        let picked = UI.with(|u| {
            let b = u.borrow();
            let ui = b.as_ref()?;
            if ui.tab != Tab::Installed {
                return None;
            }
            let e = ui.entries.get(sel)?;
            Some((e.path.clone(), e.label.clone(), ui.cfg.clone()))
        });
        let Some((path, label, cfg)) = picked else {
            set_status("open INSTALLED and select a model first");
            return;
        };

        // **Stop whatever is already running, before anything else.**
        //
        // This was missing, and it is the whole of "after one run the next model
        // does not work". Loading a second model spawned a second
        // `chaos-serve` while the first still held the port: the new process
        // died with `os error 10048`, the readiness poll got its 200 from the
        // OLD server, and the window reported the new model ready while every
        // message went to the old weights. The old child's handle was also
        // overwritten by the new one, so nothing could ever kill it -- it kept
        // its residency until the machine was rebooted, which on a 15.7 GiB box
        // running a 144 GB model is the difference between 0.45 tok/s and
        // nothing at all.
        //
        // Measured, both halves: the second server exits 10048, and `/health`
        // keeps answering with the first model's name.
        stop_server();

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

        // **Refuse a half-written container before anything is started.** An
        // interrupted download leaves a valid header and no weights, so the
        // list shows it beside models that work and the failure arrives later,
        // from the engine, in the engine's words.
        if let Some(why) = chaos_model::complete::why_incomplete(&path) {
            set_status(&format!("{label}: {why}"));
            unsafe {
                MessageBoxW(
                    main_hwnd(),
                    wide(&format!(
                        "{label} is not fully downloaded.\n\n{why}.\n\nPress DOWNLOAD, which \
                         is live for this model -- the fetch resumes from what is already on \
                         disk rather than starting again."
                    ))
                    .as_ptr(),
                    wide("Chaos").as_ptr(),
                    MB_OK | MB_ICONWARNING,
                );
            }
            return;
        }

        // **Refuse an architecture the engine cannot run, here, before a server
        // is started.** Without this the app spawned `chaos-serve`, the server
        // refused the container and exited, and the window went on showing the
        // model as RUNNING with a green dot -- so the next message came back
        // "connection actively refused", which reads as a networking fault
        // rather than as "this model does not work". That is exactly what
        // happened to Atur with Qwen3.6-27B.
        let arch = architecture_of(&path);
        if let Some(why) = arch
            .as_deref()
            .and_then(chaos_model::catalogue::why_not_runnable)
        {
            set_status(&format!("{label} cannot run: {why}"));
            unsafe {
                MessageBoxW(
                    main_hwnd(),
                    wide(&format!(
                        "{label} cannot run in Chaos yet.\n\n{why}.\n\nThe file is fine and \
                         nothing is wrong with the download -- the engine does not implement \
                         this architecture. Deleting and fetching it again will not change that."
                    ))
                    .as_ptr(),
                    wide("Chaos").as_ptr(),
                    MB_OK | MB_ICONWARNING,
                );
            }
            return;
        }

        // **A verified architecture at a size nobody diffed.** Not a refusal --
        // the model runs, and refusing something that might be fine is its own
        // kind of wrong. But the window is where somebody reads the answer, so
        // it is where the caveat has to appear: `chaos-serve` prints this on its
        // stdout, which this app does not show.
        //
        // Qwen3.6-27B is the case. It loads, generates, and prints
        // `ทัน ทัน ทัน` -- and 27B is the size Atur actually has.
        if let Some((a, n)) = arch.as_deref().zip(block_count_of(&path)) {
            if let Some(why) = chaos_model::catalogue::why_shape_is_unverified(a, n) {
                unsafe {
                    MessageBoxW(
                        main_hwnd(),
                        wide(&format!(
                            "{label} will run, but its answers are UNVERIFIED.\n\n{why}.\n\n\
                             It will load and reply normally. Whether the reply is correct has \
                             not been checked at this size, and a wrong forward pass here reads \
                             as a confident answer rather than an error."
                        ))
                        .as_ptr(),
                        wide("Chaos").as_ptr(),
                        MB_OK | MB_ICONWARNING,
                    );
                }
            }
        }

        // **`cfg.force` is whatever SETTINGS says**, and nothing here overrides
        // it. The previous version set it to `true` unconditionally, which made
        // the toggle on the settings page a decoration: turning it off changed
        // nothing, and an unverified architecture ran anyway. If it is off,
        // `chaos-serve` refuses by name and the refusal reaches the strip.
        let port = cfg.port;

        // `kill` returns before Windows has released the listening socket, so a
        // new server started immediately can still lose the bind. Wait for the
        // port to go quiet, and if something that is not ours is sitting on it,
        // say so instead of starting a process that will die.
        if let Some(other) = wait_for_port_free(port) {
            set_status(&format!(
                "port {port} is already serving {other} -- stop it, or change the port in SETTINGS"
            ));
            return;
        }

        let mut cmd = Command::new(&exe);
        for a in cfg.serve_args(&path.to_string_lossy()) {
            cmd.arg(a);
        }
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        // Kept rather than inherited, so the reason a server exits reaches the
        // window instead of a console that does not exist.
        cmd.stderr(std::process::Stdio::piped());

        match cmd.spawn() {
            Ok(mut c) => {
                let pid = c.id();
                if let Some(err) = c.stderr.take() {
                    // Drained on its own thread: a full pipe blocks the writer,
                    // which would hang the server rather than merely lose its
                    // output.
                    std::thread::spawn(move || {
                        use std::io::BufRead;
                        let mut keep: Vec<String> = Vec::new();
                        for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
                            keep.push(line);
                            // The last few lines are the ones that say why.
                            if keep.len() > 8 {
                                keep.remove(0);
                            }
                            let joined = keep.join(" ");
                            shared().lock().unwrap().serve_err = joined;
                        }
                    });
                }
                UI.with(|u| {
                    if let Some(ui) = u.borrow_mut().as_mut() {
                        ui.server = Some(c);
                        ui.history.clear();
                        ui.loaded = Some(label.clone());
                        ui.loaded_at = Some(Instant::now());
                        ui.served = 0;
                        ui.last_rate = 0.0;
                        ui.port = port;
                    }
                });
                // Borrow closed before any of this touches Windows.
                unsafe {
                    SetWindowTextW(ctl(nav::ID_OUT), wide("").as_ptr());
                }
                sync_enabled();
                // **A number, not a promise.** "loading -- a large model takes a
                // while" and then nothing is a window that looks broken for as
                // long as the load takes. The server's working set against the
                // catalogue's resident figure is the progress, measured the same
                // way a download's is: from the outside, with no cooperation
                // from the process being watched.
                let resident = catalog::resident_for(&label).unwrap_or(0);
                let mut watch = loading::Loading::new(label.clone(), resident);
                set_status(&watch.line());
                std::thread::spawn(move || {
                    let began = Instant::now();
                    // Poll rather than parse stdout: readiness is exactly "does
                    // it answer", and that is the check a client makes too.
                    //
                    // **But watch the process as well as the port.** A server
                    // that has already exited will never answer, and waiting
                    // ten minutes to conclude that -- while its one-line reason
                    // sits unread -- is how a port collision came to look like a
                    // model too slow to load.
                    for _ in 0..1200 {
                        if client::health(port) {
                            let mut s = shared().lock().unwrap();
                            s.status = "ready".into();
                            s.finished = true;
                            drop(s);
                            notify();
                            return;
                        }
                        if process_gone(pid) {
                            let mut s = shared().lock().unwrap();
                            let why = s.serve_err.trim().to_string();
                            s.status = if why.is_empty() {
                                "chaos-serve stopped before it was ready".into()
                            } else {
                                format!("chaos-serve stopped: {why}")
                            };
                            s.finished = true;
                            drop(s);
                            notify();
                            return;
                        }
                        // Sampled every turn of the same loop that checks
                        // readiness, so the line moves whenever anything else
                        // would have.
                        watch.rss = chaos_app::win32::working_set(pid).unwrap_or(watch.rss);
                        watch.elapsed = began.elapsed().as_secs_f64();
                        {
                            let mut s = shared().lock().unwrap();
                            s.status = watch.line();
                        }
                        notify();
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

    /// The architecture string in a container's header.
    ///
    /// Read straight from the GGUF rather than guessed from the filename: the
    /// name says "Qwen3.6" and the header says `qwen35`, and only one of those
    /// decides whether the engine can run it.
    fn architecture_of(path: &std::path::Path) -> Option<String> {
        use std::io::Read;
        // Only the head: a GGUF's metadata and tensor table live at the front,
        // and mapping 16 GB to read one string would stall the window. If the
        // table is longer than this the parse fails, `None` comes back, and the
        // load proceeds exactly as it did before -- the server then refuses it,
        // which is the old behaviour rather than a new failure.
        const HEAD: usize = 32 << 20;
        let mut f = std::fs::File::open(path).ok()?;
        let mut buf = Vec::with_capacity(HEAD);
        f.by_ref().take(HEAD as u64).read_to_end(&mut buf).ok()?;
        let g = chaos_gguf::Gguf::parse(&buf).ok()?;
        g.get_str("general.architecture").map(str::to_string)
    }

    /// The block count in a container's header.
    ///
    /// Beside [`architecture_of`] and reading the same 32 MiB head, because the
    /// two answers are always wanted together: the architecture says whether the
    /// engine implements this model, and the block count says whether *this
    /// size* of it was ever checked.
    fn block_count_of(path: &std::path::Path) -> Option<u32> {
        use std::io::Read;
        const HEAD: usize = 32 << 20;
        let mut f = std::fs::File::open(path).ok()?;
        let mut buf = Vec::with_capacity(HEAD);
        f.by_ref().take(HEAD as u64).read_to_end(&mut buf).ok()?;
        let g = chaos_gguf::Gguf::parse(&buf).ok()?;
        let arch = g.get_str("general.architecture")?;
        g.get_u64(&format!("{arch}.block_count")).map(|v| v as u32)
    }

    fn unload_model() {
        stop_server();
        sync_enabled();
        set_status("stopped -- the memory is back");
    }

    /// Stop the child engine, if one is running.
    ///
    /// Kill rather than ask: `chaos-serve` has no shutdown endpoint, it holds
    /// no state worth flushing, and a model mid-token would otherwise keep the
    /// process alive for minutes.
    /// Has this process exited?
    ///
    /// By handle rather than by `try_wait`, because the poll runs on a worker
    /// thread and the `Child` lives in `UI`, which is a `thread_local!` and is
    /// `None` everywhere else. A handle that cannot be opened means the process
    /// is gone, which is the answer being asked for.
    fn process_gone(pid: u32) -> bool {
        unsafe {
            let h = OpenProcess(SYNCHRONIZE, 0, pid);
            if h.is_null() {
                return true;
            }
            let gone = WaitForSingleObject(h, 0) == WAIT_OBJECT_0;
            CloseHandle(h);
            gone
        }
    }

    /// Wait for `port` to stop answering, and name whatever will not let go.
    ///
    /// Returns `None` once the port is free, or the model name still serving on
    /// it. Five seconds is generous: this is the gap between `TerminateProcess`
    /// returning and Windows releasing the socket, which is milliseconds.
    fn wait_for_port_free(port: u16) -> Option<String> {
        for _ in 0..20 {
            client::health_model(port)?;
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        // Still answering. Name it if it said who it is.
        client::health_model(port).map(|n| {
            if n.is_empty() {
                "another server".into()
            } else {
                n
            }
        })
    }

    fn stop_server() {
        let child = UI.with(|u| u.borrow_mut().as_mut().and_then(|ui| ui.server.take()));
        if let Some(mut c) = child {
            let _ = c.kill();
            let _ = c.wait();
        }
        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                ui.loaded = None;
                ui.loaded_at = None;
            }
        });
    }

    /// Every file belonging to a container, including the other shards.
    ///
    /// Deleting only the file the list shows would leave four of five shards --
    /// 120 GB of unusable data -- on disk, and report success.
    fn shards_of(first: &std::path::Path) -> Vec<std::path::PathBuf> {
        let single = || vec![first.to_path_buf()];
        let Some(name) = first.file_name().and_then(|n| n.to_str()) else {
            return single();
        };
        let Some(dir) = first.parent() else {
            return single();
        };
        let Some(idx) = name.rfind("-00001-of-") else {
            return single();
        };
        let stem = &name[..idx];
        let mut out: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(stem) && n.ends_with(".gguf"))
            })
            .collect();
        out.sort();
        if out.is_empty() {
            out = single();
        }
        out
    }

    /// Delete the selected model's files, after saying exactly what will go.
    fn delete_selected() {
        let Some(sel) = selection() else {
            set_status("select an installed model to delete");
            return;
        };
        let picked = UI.with(|u| {
            let b = u.borrow();
            let ui = b.as_ref()?;
            if ui.tab != Tab::Installed {
                return None;
            }
            let e = ui.entries.get(sel)?;
            Some((
                e.path.clone(),
                e.label.clone(),
                e.bytes.unwrap_or(0),
                ui.loaded.clone(),
            ))
        });
        let Some((path, label, bytes, loaded)) = picked else {
            set_status("select an installed model to delete");
            return;
        };
        // Deleting the file a running server has open would half-work and leave
        // the engine reading a hole.
        if loaded.as_deref() == Some(label.as_str()) {
            set_status("that model is running -- press STOP first");
            return;
        }

        let files = shards_of(&path);
        let msg = format!(
            "Delete {label}?\n\n{} file(s), {} freed from\n{}\n\nThis cannot be undone.",
            files.len(),
            models::human_size(bytes),
            path.parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        );
        let answer = unsafe {
            MessageBoxW(
                main_hwnd(),
                wide(&msg).as_ptr(),
                wide("Chaos").as_ptr(),
                MB_YESNO | MB_ICONWARNING,
            )
        };
        if answer != IDYES {
            set_status("nothing deleted");
            return;
        }

        let mut gone = 0usize;
        let mut failed: Vec<String> = Vec::new();
        for f in &files {
            match std::fs::remove_file(f) {
                Ok(()) => gone += 1,
                Err(e) => failed.push(format!("{}: {e}", f.display())),
            }
        }
        if failed.is_empty() {
            set_status(&format!(
                "deleted {label} -- {gone} file(s), {} freed",
                models::human_size(bytes)
            ));
        } else {
            set_status(&format!(
                "deleted {gone}, {} failed: {}",
                failed.len(),
                failed[0]
            ));
        }
        rescan();
    }

    /// Fetch the selected catalogue entry with `chaos-pull`.
    ///
    /// A child process again, for the same reason as the server: `chaos-pull`
    /// already knows how to resume, verify and place a five-shard container,
    /// and a second downloader in the window would be a second thing to get
    /// wrong about a 155 GB file.
    fn download_selected() {
        let Some(sel) = selection() else {
            set_status("select something to download");
            return;
        };
        let offer = UI.with(|u| {
            let b = u.borrow();
            let ui = b.as_ref()?;
            if ui.tab == Tab::Available {
                return ui
                    .offers
                    .get(sel)
                    .map(|o| (o.name.clone(), o.quant.clone(), o.bytes));
            }
            // **On INSTALLED this is the resume button.** A model already on
            // disk is known by its file rather than by the name it was fetched
            // under, so the catalogue is asked which entry produces that
            // filename. `chaos-pull` then resumes from the bytes already there.
            let e = ui.entries.get(sel)?;
            e.incomplete.as_ref()?;
            let file = e.path.file_name()?.to_str()?;
            let (entry, quant) = chaos_model::catalogue::find_by_file(file)?;
            Some((entry.name.to_string(), quant.name.to_string(), quant.bytes))
        });
        let (Some((name, quant, bytes)), Some(exe)) = (offer, std::env::current_exe().ok()) else {
            set_status("open AVAILABLE and select something to download");
            return;
        };
        let pull = exe.with_file_name("chaos-pull.exe");
        if !pull.exists() {
            set_status("chaos-pull.exe is missing from this folder");
            return;
        }
        // **The age gate, in the window rather than in the helper.**
        //
        // `chaos-pull` asks at a terminal, and the app spawns it with
        // `CREATE_NO_WINDOW` -- so its prompt read EOF, cancelled, and returned
        // success, and the window said "downloaded" for a file that was never
        // fetched. The dialog belongs here, where there is somebody to answer it.
        let adult = chaos_model::catalogue::find(&name).is_some_and(|e| e.adult);
        if adult {
            let answer = unsafe {
                MessageBoxW(
                    main_hwnd(),
                    wide(&format!(
                        "{name} is an ADULT model, published for generating explicit \
                         imagery.\n\nChaos does not filter what a model produces.\n\n\
                         Are you at least 18 years old, and is adult material lawful \
                         where you are?"
                    ))
                    .as_ptr(),
                    wide("Adult content -- 18+").as_ptr(),
                    MB_YESNO | MB_ICONWARNING,
                )
            };
            if answer != IDYES {
                set_status(&format!("{name} not downloaded: age not confirmed"));
                return;
            }
        }

        let dir = models::default_dir();
        let _ = std::fs::create_dir_all(&dir);

        // Everything the watcher needs, worked out before the fetch starts:
        // which files will appear, and what they will weigh in total.
        let files: Vec<std::path::PathBuf> = chaos_model::catalogue::find(&name)
            .and_then(|e| e.quant(&quant).map(|q| (e, q)))
            // `local_name`, matching where `chaos-pull` actually writes: the
            // watcher looked for `split_files/vae/x.safetensors` under the models
            // folder and would have waited forever for a file saved as `x`.
            .map(|(e, q)| {
                e.files(q)
                    .into_iter()
                    .map(|f| dir.join(chaos_model::catalogue::Entry::local_name(&f)))
                    .collect()
            })
            .unwrap_or_default();
        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                ui.download = Some(Download::new(format!("{name} {quant}"), files, bytes));
            }
        });
        set_status(&format!(
            "downloading {name} {quant}, {}",
            models::human_size(bytes)
        ));

        std::thread::spawn(move || {
            let mut cmd = Command::new(&pull);
            cmd.arg(&name)
                .arg("--quant")
                .arg(&quant)
                .arg("--dir")
                .arg(&dir)
                .arg("--yes");
            // Consent from the dialog above, passed on. Set only when a person
            // clicked Yes; `--yes` above means "do not ask about the size" and
            // has never meant this.
            if adult {
                cmd.env("CHAOS_ADULT_CONFIRMED", "1");
            }
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(CREATE_NO_WINDOW);
            }
            let msg = match cmd.status() {
                Ok(st) if st.success() => format!("{name} {quant} downloaded"),
                // 3 is "the age check was not satisfied", which is not a
                // failure to report as one.
                Ok(st) if st.code() == Some(3) => {
                    format!("{name} {quant} not downloaded: age not confirmed")
                }
                Ok(st) => format!("download failed (exit {})", st.code().unwrap_or(-1)),
                Err(e) => format!("could not start chaos-pull: {e}"),
            };
            let mut sh = shared().lock().unwrap();
            sh.status = msg;
            sh.finished = true;
            sh.download_done = true;
            drop(sh);
            notify();
        });
    }

    fn endpoint() -> Option<String> {
        UI.with(|u| {
            let b = u.borrow();
            let ui = b.as_ref()?;
            ui.loaded
                .as_ref()
                .map(|_| format!("http://127.0.0.1:{}/v1", ui.port))
        })
    }

    /// Put the endpoint on the clipboard.
    ///
    /// This is the string you paste into a coding agent, and retyping it off
    /// the screen is exactly the friction that makes a window feel like a demo.
    fn copy_endpoint() {
        let Some(url) = endpoint() else {
            set_status("nothing is running, so there is no endpoint yet");
            return;
        };
        let key = UI.with(|u| u.borrow().as_ref().and_then(|ui| ui.cfg.api_key.clone()));
        // With a key set, both go on the clipboard. Pasting a URL into a client
        // that then rejects every request for want of a key is the kind of
        // half-answer that costs an afternoon.
        let text = match &key {
            Some(k) => format!(
                "{url}
API key: {k}"
            ),
            None => url.clone(),
        };
        if unsafe { set_clipboard_text(main_hwnd(), &text) } {
            match key {
                Some(_) => set_status(&format!("copied {url} and its API key")),
                None => set_status(&format!("copied {url}")),
            }
        } else {
            set_status("the clipboard is held by another program");
        }
    }

    /// Turn the API key on or off, generating one when turning it on.
    ///
    /// **Off by default, and never switched on quietly.** Enabling it would
    /// start refusing every agent already pointed at this endpoint, so it is a
    /// deliberate act that shows the key, copies it, and says when it takes
    /// effect -- a server that is already running was started without it.
    fn toggle_api_key() {
        let key = UI.with(|u| {
            let mut b = u.borrow_mut();
            let ui = b.as_mut()?;
            ui.cfg.api_key = match ui.cfg.api_key {
                Some(_) => None,
                // 24 bytes from the system generator, hex. **Not derived from
                // the clock**: a key anyone can guess from roughly when it was
                // made is worse than no key, because it is trusted.
                None => random_hex(24),
            };
            let _ = ui.cfg.save();
            Some(ui.cfg.api_key.clone())
        });
        let Some(key) = key else { return };
        let running = UI.with(|u| u.borrow().as_ref().is_some_and(|ui| ui.loaded.is_some()));
        let msg = match &key {
            Some(k) => {
                unsafe {
                    set_clipboard_text(main_hwnd(), k);
                }
                format!(
                    "An API key is now required.

{k}

It is on your clipboard.                      Paste it into any client that asks for one.

{}",
                    if running {
                        "The running model was started without it -- press STOP and LOAD again."
                    } else {
                        "It takes effect the next time you load a model."
                    }
                )
            }
            None => "No API key is required now.

Any value a client sends is accepted.                      The server still listens on 127.0.0.1 only."
                .to_string(),
        };
        unsafe {
            MessageBoxW(
                main_hwnd(),
                wide(&msg).as_ptr(),
                wide("Chaos").as_ptr(),
                MB_OK | MB_ICONINFORMATION,
            );
        }
        repaint();
    }

    /// Do what a coding agent does, and say what happened.
    ///
    /// **"Point your agent at this URL" is advice, not evidence.** A user whose
    /// agent will not connect has no way to tell whether the fault is the port,
    /// the key, the model or the client, and the usual answer -- try `curl` --
    /// is not one most people have. This runs the three requests an
    /// OpenAI-compatible client makes and reports each.
    fn test_connection() {
        let (port, key, running) = UI.with(|u| {
            let b = u.borrow();
            b.as_ref()
                .map(|ui| (ui.port, ui.cfg.api_key.clone(), ui.loaded.is_some()))
                .unwrap_or((0, None, false))
        });
        if !running {
            set_status("load a model first -- there is nothing to connect to");
            show_page(Page::Models);
            return;
        }
        set_status("testing the connection...");
        // On a worker: the completion it makes is a real one, and on a large
        // model that is seconds. Freezing the window to prove it is not frozen
        // would be its own joke.
        std::thread::spawn(move || {
            let checks = client::check(port, key.as_deref());
            let all = checks.iter().all(|c| c.ok);
            let mut msg = String::new();
            for c in &checks {
                msg.push_str(&format!(
                    "{}  {}
    {}

",
                    if c.ok { "[ ok ]" } else { "[fail]" },
                    c.name,
                    c.detail
                ));
            }
            msg.push_str(&if all {
                format!(
                    "Any OpenAI-compatible agent will work. Paste these into it:\n\n\
                     Base URL   http://127.0.0.1:{port}/v1\n\
                     API key    {}\n\n\
                     In Hermes: provider \"custom\", and that base URL.",
                    key.as_deref().unwrap_or("not required -- send any value")
                )
            } else {
                "Something in the chain is not answering. The failing line above                  says which."
                    .to_string()
            });
            let mut sh = shared().lock().unwrap();
            sh.status = if all {
                "connection test passed".into()
            } else {
                "connection test failed".into()
            };
            // Handed to the UI thread rather than shown here; see `Shared`.
            sh.report = Some((msg, all));
            drop(sh);
            notify();
        });
    }

    fn send_prompt() {
        if busy().load(Ordering::SeqCst) {
            return;
        }
        let running = UI.with(|u| u.borrow().as_ref().is_some_and(|ui| ui.loaded.is_some()));
        if !running {
            // Say what to do, and go there. A SEND that silently does nothing
            // is how "why can I not send a message" starts.
            set_status("nothing is running -- pick a model and press LOAD");
            show_page(Page::Models);
            return;
        }
        let prompt = control_text(ctl(nav::ID_IN)).trim().to_string();
        if prompt.is_empty() {
            return;
        }
        let (port, history, key) = UI.with(|u| {
            let b = u.borrow();
            b.as_ref()
                .map(|ui| (ui.port, ui.history.clone(), ui.cfg.api_key.clone()))
                .unwrap_or((0, Vec::new(), None))
        });

        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                ui.answer.clear();
                ui.history.push(("user".into(), prompt.clone()));
            }
        });
        unsafe {
            SetWindowTextW(ctl(nav::ID_IN), wide("").as_ptr());
            EnableWindow(ctl(nav::ID_SEND), 0);
        }
        append_out(&format!("\r\n> {}\r\n\r\n", prompt.replace('\n', "\r\n")));

        {
            let mut s = shared().lock().unwrap();
            s.pending.clear();
            s.finished = false;
            s.tokens = 0;
            s.started = Some(Instant::now());
            s.status = "thinking".into();
        }
        busy().store(true, Ordering::SeqCst);

        std::thread::spawn(move || {
            client::chat(
                port,
                &history,
                &prompt,
                512,
                key.as_deref(),
                &mut |ev| match ev {
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
                },
            );
        });
    }

    /// Empty the transcript and the history behind it.
    fn clear_chat() {
        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                ui.history.clear();
                ui.answer.clear();
            }
        });
        unsafe {
            SetWindowTextW(ctl(nav::ID_OUT), wide("").as_ptr());
        }
        set_status("conversation cleared");
    }

    /// Drain what the worker produced. Runs on the UI thread only.
    fn drain() {
        let (text, finished, tokens, elapsed, report) = {
            let mut s = shared().lock().unwrap();
            let t = std::mem::take(&mut s.pending);
            let e = s.started.map(|i| i.elapsed().as_secs_f64()).unwrap_or(0.0);
            (t, s.finished, s.tokens, e, s.report.take())
        };
        // A report a worker produced, written into the transcript rather than
        // put up as a message box.
        //
        // **A modal box was the first attempt and it was the wrong tool twice
        // over.** Shown from the worker it never appeared at all -- an owner
        // window belonging to another thread is undefined and here does
        // nothing. Moved to this thread it works, but a connection report is
        // exactly the text somebody needs to copy into a bug report or an agent
        // config, and a message box is the one place Windows will not let you
        // select from. The transcript is selectable, scrollable and already on
        // screen.
        if let Some((msg, _good)) = report {
            // An EDIT wants CRLF; a bare LF renders as a box.
            let body = msg.replace("\r\n", "\n").replace('\n', "\r\n");
            append_out(&format!("\r\n{body}\r\n"));
            show_page(Page::Chat);
        }
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
            UI.with(|u| {
                if let Some(ui) = u.borrow_mut().as_mut() {
                    ui.last_rate = rate;
                }
            });
        }
        show_update_outcome();
        // The installer is up; get out of its way. Anything running under this
        // window goes with it, which is what the installer needs anyway.
        if shared().lock().unwrap().update_quit {
            let h = main_hwnd();
            if !h.is_null() {
                unsafe {
                    PostMessageW(h, WM_CLOSE, 0, 0);
                }
            }
        }
        if finished && busy().swap(false, Ordering::SeqCst) {
            UI.with(|u| {
                if let Some(ui) = u.borrow_mut().as_mut() {
                    let a = std::mem::take(&mut ui.answer);
                    if !a.is_empty() {
                        ui.history.push(("assistant".into(), a));
                    }
                    ui.served += u64::from(tokens);
                }
            });
            sync_enabled();
        }
        repaint();
    }

    // -- settings ------------------------------------------------------------

    /// The list a field offers on this machine, or `None` if it is typed.
    fn field_choices(id: i32) -> Option<Vec<Choice>> {
        let m = UI.with(|u| u.borrow().as_ref().map(|ui| ui.machine))?;
        choices::for_field(id, m)
    }

    /// Put the stored settings into the boxes and the dropdowns.
    fn fill_settings_page() {
        let cfg = UI.with(|u| u.borrow().as_ref().map(|ui| ui.cfg.clone()));
        let Some(cfg) = cfg else { return };

        let stored = |id: i32| -> String {
            match id {
                nav::ID_CACHE => cfg.cache_gib.map(|v| format!("{v}")).unwrap_or_default(),
                nav::ID_THREADS => cfg.threads.map(|v| v.to_string()).unwrap_or_default(),
                nav::ID_THREADS_BATCH => {
                    cfg.threads_batch.map(|v| v.to_string()).unwrap_or_default()
                }
                nav::ID_CONTEXT => cfg.context.map(|v| v.to_string()).unwrap_or_default(),
                nav::ID_NGL => cfg.ngl.map(|v| v.to_string()).unwrap_or_default(),
                nav::ID_PORT => cfg.port.to_string(),
                nav::ID_MODELS_DIR => cfg.models_dir.clone().unwrap_or_default(),
                _ => String::new(),
            }
        };

        for f in nav::FIELDS {
            let h = ctl(f.id);
            if h.is_null() {
                continue;
            }
            let want = stored(f.id);
            match field_choices(f.id) {
                Some(list) => {
                    // A value the file holds that is not in the list still has
                    // to be selectable, or saving would silently change it.
                    let mut list = list;
                    if !want.is_empty() && !list.iter().any(|c| c.value == want) {
                        list.push(Choice {
                            value: want.clone(),
                            label: format!("{want} (from your settings file)"),
                            note: "Typed by hand rather than chosen here. Kept as it is.".into(),
                        });
                    }
                    let selected = list.iter().position(|c| c.value == want).unwrap_or(0);
                    unsafe {
                        SendMessageW(h, CB_RESETCONTENT, 0, 0);
                        for c in &list {
                            let t = wide(&c.label);
                            SendMessageW(h, CB_ADDSTRING, 0, t.as_ptr() as LPARAM);
                        }
                        SendMessageW(h, CB_SETCURSEL, selected, 0);
                        widen_dropdown(h, &list);
                    }
                    // Cached so paint and save do not rebuild the list, and so
                    // the note under the box follows the selection.
                    UI.with(|u| {
                        if let Some(ui) = u.borrow_mut().as_mut() {
                            ui.lists.insert(f.id, list);
                        }
                    });
                }
                None => unsafe {
                    SetWindowTextW(h, wide(&want).as_ptr());
                },
            }
        }
    }

    /// Make the open list as wide as its longest option.
    ///
    /// **A drop-down is exactly as wide as its box unless it is told
    /// otherwise**, and these boxes sit in a settings column narrower than the
    /// sentences in them -- so "Processor (the GPU is not used here)" opened as
    /// "Processor (the GPU is not used her...", with the ellipsis landing on
    /// the part that says what the option does. Atur's report was that the
    /// drop-downs did not work well, and this is the whole of it: they opened,
    /// they selected, they could not be read.
    ///
    /// Measured against the same font the rows are drawn in, capped so one long
    /// label cannot push the list off the screen.
    unsafe fn widen_dropdown(h: HWND, list: &[Choice]) {
        let hdc = GetDC(h);
        if hdc.is_null() {
            return;
        }
        let font = UI.with(|u| u.borrow().as_ref().map(|ui| ui.fonts.body));
        let Some(font) = font else {
            ReleaseDC(h, hdc);
            return;
        };
        let widest = list
            .iter()
            .map(|c| text_width(hdc, &c.label, font))
            .max()
            .unwrap_or(0);
        ReleaseDC(h, hdc);

        let mut r = RECT::default();
        GetWindowRect(h, &mut r);
        let own = r.right - r.left;
        // 10px of padding either side in `draw_combo`, plus room for the
        // scrollbar the list grows when there are more than `COMBO_VISIBLE`.
        let want = (widest + 20 + 20).max(own);
        // Never wider than the screen it opens on.
        let cap = work_area().map(|(l, _, r, _)| r - l).unwrap_or(1280) - 80;
        SendMessageW(h, CB_SETDROPPEDWIDTH, want.min(cap) as WPARAM, 0);
    }

    /// The value a settings control currently holds, whichever kind it is.
    fn field_value(id: i32) -> String {
        let h = ctl(id);
        if h.is_null() {
            return String::new();
        }
        let listed = UI.with(|u| {
            let b = u.borrow();
            b.as_ref()
                .map(|ui| ui.lists.contains_key(&id))
                .unwrap_or(false)
        });
        if !listed {
            return control_text(h).trim().to_string();
        }
        let sel = unsafe { SendMessageW(h, CB_GETCURSEL, 0, 0) };
        if sel < 0 {
            return String::new();
        }
        UI.with(|u| {
            let b = u.borrow();
            b.as_ref()
                .and_then(|ui| ui.lists.get(&id))
                .and_then(|l| l.get(sel as usize))
                .map(|c| c.value.clone())
                .unwrap_or_default()
        })
    }

    /// Read every box back and write the file.
    /// Pick the models folder instead of typing it.
    ///
    /// **The chosen folder is appended, not substituted, when one is already
    /// set.** The field takes several folders separated by `;` and people do
    /// use that -- a fast disk and an archive drive -- so replacing the lot
    /// because somebody wanted to add a second one would quietly lose the
    /// first. Picking the folder that is already there changes nothing.
    fn browse_models_dir(hwnd: HWND) {
        let current = field_value(nav::ID_MODELS_DIR);
        let first = current.split(';').next().unwrap_or("").trim().to_string();
        let start = if first.is_empty() {
            models::default_dir().to_string_lossy().to_string()
        } else {
            first
        };
        let Some(picked) =
            chaos_app::win32::pick_folder(hwnd, "Where models are kept", Some(&start))
        else {
            return;
        };
        let already: Vec<&str> = current
            .split(';')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .collect();
        let next = if already.iter().any(|p| p.eq_ignore_ascii_case(&picked)) {
            current.clone()
        } else if already.is_empty() {
            picked.clone()
        } else {
            format!("{};{}", already.join(";"), picked)
        };
        unsafe {
            SetWindowTextW(ctl(nav::ID_MODELS_DIR), wide(&next).as_ptr());
        }
        set_status(&format!("models folder: {next} -- press SAVE to keep it"));
    }

    fn save_settings() {
        // Reads a dropdown's selected *value* or a box's text, as appropriate.
        let read = field_value;
        let cache = read(nav::ID_CACHE);
        let threads = read(nav::ID_THREADS);
        let tb = read(nav::ID_THREADS_BATCH);
        let port_text = read(nav::ID_PORT);
        let context = read(nav::ID_CONTEXT);
        let ngl = read(nav::ID_NGL);
        let dir = read(nav::ID_MODELS_DIR);

        // A port that will not parse must not silently become something else:
        // the endpoint the window advertises has to be the port the server
        // binds, or every client is sent to nothing.
        let Some(port) = port_text.parse::<u16>().ok().filter(|p| *p > 0) else {
            set_status(&format!("{port_text:?} is not a port -- nothing saved"));
            return;
        };

        let result = UI.with(|u| {
            let mut b = u.borrow_mut();
            let ui = b.as_mut()?;
            ui.cfg.cache_gib = cache.parse::<f64>().ok().filter(|v| *v > 0.0);
            ui.cfg.threads = threads.parse::<u32>().ok().filter(|v| *v > 0);
            ui.cfg.threads_batch = tb.parse::<u32>().ok().filter(|v| *v > 0);
            ui.cfg.port = port;
            ui.cfg.context = context.parse::<u32>().ok().filter(|v| *v > 0);
            ui.cfg.ngl = ngl.parse::<u32>().ok();
            ui.cfg.models_dir = (!dir.is_empty()).then(|| dir.clone());
            Some((ui.cfg.save(), ui.loaded.is_some()))
        });
        let Some((saved, running)) = result else {
            return;
        };
        match saved {
            // The port only reaches a running server on the next load; saying
            // so beats letting someone wonder why the endpoint did not move.
            Ok(()) if running => set_status("saved -- the port takes effect next time you load"),
            Ok(()) => set_status(&format!("saved to {}", settings::path().display())),
            Err(e) => set_status(&e),
        }
        // What was stored is what is shown: re-filling normalises "8231 " and
        // reveals anything dropped for being unparseable.
        fill_settings_page();
        repaint();
    }

    /// Back to measured everything, which is what an empty file means.
    fn reset_settings() {
        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                // The theme is a view preference, not an engine setting, and
                // resetting the engine must not flip the lights.
                ui.cfg.reset_engine();
                let _ = ui.cfg.save();
            }
        });
        fill_settings_page();
        set_status("reset -- every setting is measured from this machine again");
        repaint();
    }

    fn toggle(id: i32) {
        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                if id == nav::ID_AUTO {
                    ui.cfg.auto = !ui.cfg.auto;
                } else {
                    ui.cfg.force = !ui.cfg.force;
                }
                let _ = ui.cfg.save();
            }
        });
        repaint();
    }

    // -- painting primitives -------------------------------------------------

    /// One place that draws text, so no page invents its own alignment.
    unsafe fn text(hdc: HDC, r: RECT, s: &str, font: HFONT, colour: Rgb, flags: u32) {
        let old = SelectObject(hdc, font as HGDIOBJ);
        SetTextColor(hdc, colour);
        SetBkMode(hdc, TRANSPARENT);
        let mut rc = r;
        let w: Vec<u16> = s.encode_utf16().collect();
        // **Never hand Windows an empty Vec's pointer.** `Vec::as_ptr` on an
        // empty vector returns a dangling (aligned but unallocated) address and
        // `DrawTextW` dereferences it. That killed the installer outright the
        // moment its report reached a blank line -- a stack-cookie failure with
        // no panic message, because a fault inside an `extern "system"` call
        // never reaches the panic hook. There is nothing to draw either way.
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

    /// Left-aligned single line, which is nearly every string in the app.
    unsafe fn label(hdc: HDC, x: i32, y: i32, w: i32, s: &str, font: HFONT, colour: Rgb) {
        text(
            hdc,
            RECT {
                left: x,
                top: y,
                right: x + w,
                bottom: y + 40,
            },
            s,
            font,
            colour,
            DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS,
        );
    }

    unsafe fn fill(hdc: HDC, r: RECT, colour: Rgb) {
        let b = CreateSolidBrush(colour);
        FillRect(hdc, &r, b);
        DeleteObject(b as HGDIOBJ);
    }

    /// A hairline. The only divider this design has -- Hermes: *"group with
    /// whitespace and a single hairline, never nested rounded boxes"*.
    unsafe fn rule(hdc: HDC, x1: i32, x2: i32, y: i32, colour: Rgb) {
        fill(
            hdc,
            RECT {
                left: x1,
                top: y,
                right: x2,
                bottom: y + 1,
            },
            colour,
        );
    }

    /// A one-pixel rectangle with nothing inside it.
    unsafe fn frame(hdc: HDC, r: RECT, colour: Rgb) {
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
            fill(hdc, q, colour);
        }
    }

    fn inset(r: RECT, by: i32) -> RECT {
        RECT {
            left: r.left + by,
            top: r.top + by,
            right: r.right - by,
            bottom: r.bottom - by,
        }
    }

    unsafe fn measure(hdc: HDC, s: &str, font: HFONT) -> i32 {
        let old = SelectObject(hdc, font as HGDIOBJ);
        let w: Vec<u16> = s.encode_utf16().collect();
        let mut sz = SIZE::default();
        GetTextExtentPoint32W(hdc, w.as_ptr(), w.len() as i32, &mut sz);
        SelectObject(hdc, old);
        sz.cx
    }

    /// `a` with `pct` percent of `b` mixed in, channel by channel.
    fn mix(a: Rgb, b: Rgb, pct: u32) -> Rgb {
        let ch = |shift: u32| {
            let (x, y) = ((a >> shift) & 0xFF, (b >> shift) & 0xFF);
            ((x * (100 - pct) + y * pct) / 100) & 0xFF
        };
        ch(0) | (ch(8) << 8) | (ch(16) << 16)
    }

    fn human_duration(secs: u64) -> String {
        match secs {
            0..=59 => format!("{secs}s"),
            60..=3599 => format!("{}m {}s", secs / 60, secs % 60),
            _ => format!("{}h {}m", secs / 3600, (secs % 3600) / 60),
        }
    }

    // -- geometry, shared by layout and paint --------------------------------

    /// The area a page owns: everything but the rail and the strip.
    fn page_rect(client: RECT) -> RECT {
        RECT {
            left: metric::RAIL,
            top: 0,
            right: client.right,
            bottom: client.bottom - metric::STRIP,
        }
    }

    /// Where a navigation button sits.
    fn nav_rect(i: usize) -> RECT {
        let y = 112 + i as i32 * (metric::NAV_ROW + 2);
        RECT {
            left: 10,
            top: y,
            right: metric::RAIL - 10,
            bottom: y + metric::NAV_ROW,
        }
    }

    /// Where the model list ends and the model's own page begins.
    fn models_split(page: RECT) -> i32 {
        let usable = page.right - page.left - metric::INSET * 2;
        page.left + metric::INSET + (usable * 54 / 100)
    }

    /// The first line of a page's content, under the title block.
    fn content_top(page: RECT) -> i32 {
        page.top + 104
    }

    fn settings_columns(page: RECT) -> (i32, i32, i32) {
        let x = page.left + metric::INSET;
        let w = page.right - x - metric::INSET;
        let col = (w - 44) / 2;
        (x, x + col + 44, col)
    }

    /// One entry per settings control: its id, where it goes, and whether it is
    /// a toggle rather than a box.
    ///
    /// **Computed in one place** because `layout` positions the controls and
    /// `paint` draws the labels and hints beside them. Two independent walks of
    /// the same list is exactly how a label ends up over the wrong box.
    /// Extra height a settings field needs beyond label, control and note.
    ///
    /// Only the models folder, which has BROWSE under it.
    ///
    /// **Both walkers ask this, because drifting apart is exactly what went
    /// wrong.** `settings_rows` put BROWSE six pixels under the field and
    /// `paint_settings` painted the field's note in the same six pixels, so the
    /// button sat on top of the sentence that explains what the field is for --
    /// "%\.chaos\models. Several, separated by ; are all searched." read as
    /// "...chaos\models. Several, separated by ; are all searched." with its
    /// first half behind a button. Two walkers stepping by different amounts is
    /// the shape of that bug, and one shared step is the fix.
    fn field_extra(id: i32) -> i32 {
        if id == nav::ID_MODELS_DIR {
            metric::BUTTON + 12
        } else {
            0
        }
    }

    fn settings_rows(page: RECT) -> Vec<(i32, i32, i32, bool)> {
        let (left, right, _) = settings_columns(page);
        let mut out = Vec::new();
        let mut y = [content_top(page), content_top(page)];
        for (gi, group) in nav::GROUPS.iter().enumerate() {
            // Groups alternate between the columns, each kept whole.
            let c = usize::from(gi % 2 == 1);
            let x = if c == 0 { left } else { right };
            y[c] += 26;
            for f in nav::FIELDS.iter().filter(|f| f.group == *group) {
                out.push((f.id, x, y[c] + 18, false));
                y[c] += 18 + metric::CONTROL + 34 + field_extra(f.id);
            }
            for f in nav::TOGGLES.iter().filter(|f| f.group == *group) {
                out.push((f.id, x, y[c], true));
                y[c] += 24 + 30;
            }
            y[c] += 16;
        }
        out
    }

    // -- painting ------------------------------------------------------------

    unsafe fn paint(hwnd: HWND) {
        let mut ps: PAINTSTRUCT = std::mem::zeroed();
        let hdc = BeginPaint(hwnd, &mut ps);
        let mut r = RECT::default();
        GetClientRect(hwnd, &mut r);

        // The selection is read *before* the borrow: `LB_GETCURSEL` is a
        // `SendMessageW`, and this file's rule is that Windows is never called
        // with `UI` held open.
        let sel = selection();

        // Double-buffered. Painting a rail, a page and a strip straight to the
        // window flickers, and the transcript repaints on every token.
        let mem = CreateCompatibleDC(hdc);
        let bmp = CreateCompatibleBitmap(hdc, r.right.max(1), r.bottom.max(1));
        let old_bmp = SelectObject(mem, bmp);

        UI.with(|u| {
            let b = u.borrow();
            let Some(ui) = b.as_ref() else { return };
            fill(mem, r, ui.theme.bg);
            paint_rail(mem, ui, r);
            paint_strip(mem, ui, r);

            let page = page_rect(r);
            match ui.page {
                Page::Chat => paint_chat(mem, ui, page),
                Page::Models => paint_models(mem, ui, page, sel),
                Page::Monitor => paint_monitor(mem, ui, page),
                Page::Settings => paint_settings(mem, ui, page),
            }
        });

        BitBlt(hdc, 0, 0, r.right, r.bottom, mem, 0, 0, SRCCOPY);
        SelectObject(mem, old_bmp);
        DeleteObject(bmp);
        DeleteDC(mem);
        EndPaint(hwnd, &ps);
    }

    /// The navigation rail: the mark, the destinations, and nothing else.
    unsafe fn paint_rail(hdc: HDC, ui: &Ui, client: RECT) {
        let t = &ui.theme;
        fill(
            hdc,
            RECT {
                left: 0,
                top: 0,
                right: metric::RAIL,
                bottom: client.bottom,
            },
            t.chrome,
        );
        // One hairline separates the rail from the page. Not a border.
        fill(
            hdc,
            RECT {
                left: metric::RAIL - 1,
                top: 0,
                right: metric::RAIL,
                bottom: client.bottom,
            },
            t.stroke_3,
        );

        // The mark, box-filtered from the 256px master and blended in the
        // theme's own foreground.
        //
        // **Not the accent.** Drawing it blue was a mistake: `#0000F2` is the
        // brand's *ground* -- what Hermes puts behind its wordmark -- and the
        // logo itself is black art. And not `StretchDIBits` over a 1-bit 56px
        // bitmap either, which is what made it a blob at 30 pixels.
        // **64, not 44 and certainly not 32.** This mark is a sun of two dozen
        // fine rays around an eye, and a ray is roughly one pixel wide at 44 --
        // so the rays alias against each other and the eye is barely there.
        // Atur reported "low quality logo on top" at 44 as he had at 32. The
        // art is scan-converted from outlines at whatever size is asked for
        // (`art::logo_coverage`) and cached, so a larger box costs rail width
        // and nothing else: the rail is 208px, the mark starts at INSET and the
        // wordmark follows it, which still leaves room for five letters.
        let box_px = 64usize;
        let cov = art::logo_scaled(box_px);
        let chan = |c: Rgb, shift: u32| ((c >> shift) & 0xFF) as i32;
        let mut px = vec![0u8; box_px * box_px * 4];
        for y in 0..box_px {
            for x in 0..box_px {
                // A DIB with a positive height is bottom-up, so the source row
                // is mirrored here rather than the image being upside down.
                let a = i32::from(cov[(box_px - 1 - y) * box_px + x]);
                let i = (y * box_px + x) * 4;
                // A DIB is BGRA; `Rgb` is 0x00BBGGRR, so blue is the colour's
                // high byte and the pixel's low one.
                for (o, shift) in [(0usize, 16u32), (1, 8), (2, 0)] {
                    let (fg, bg) = (chan(t.fg, shift), chan(t.chrome, shift));
                    px[i + o] = (bg + (fg - bg) * a / 255) as u8;
                }
                px[i + 3] = 0;
            }
        }
        let bmi = BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: box_px as i32,
            biHeight: box_px as i32,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB,
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        };
        let box_px = box_px as i32;
        // Blitted 1:1, because the filtering already happened at the right size.
        StretchDIBits(
            hdc,
            metric::INSET,
            24,
            box_px,
            box_px,
            0,
            0,
            box_px,
            box_px,
            px.as_ptr() as *const std::ffi::c_void,
            &bmi,
            DIB_RGB_COLORS,
            SRCCOPY,
        );
        label(
            hdc,
            metric::INSET + box_px + 10,
            44,
            metric::RAIL,
            "CHAOS",
            ui.fonts.mark,
            t.fg,
        );
        // The version, quietly, so a bug report can name it.
        label(
            hdc,
            metric::INSET,
            92,
            metric::RAIL - metric::INSET,
            &format!("v{}", env!("CARGO_PKG_VERSION")),
            ui.fonts.small,
            t.fg_tertiary,
        );
    }

    /// The strip along the bottom: what is running, where, and how fast.
    ///
    /// **The only thing on every page.** Whatever you are looking at, this says
    /// whether a model is up and gives you the URL to paste.
    unsafe fn paint_strip(hdc: HDC, ui: &Ui, client: RECT) {
        let t = &ui.theme;
        let top = client.bottom - metric::STRIP;
        fill(
            hdc,
            RECT {
                left: 0,
                top,
                right: client.right,
                bottom: client.bottom,
            },
            t.chrome,
        );
        rule(hdc, 0, client.right, top, t.stroke_3);

        let y = top + 8;
        let x = metric::RAIL + metric::INSET;

        // **A download outranks the running model on the strip.** It is the
        // thing that is happening, it can take an hour, and a window that says
        // only "downloading" for that hour looks broken -- which is what it did.
        if let Some(d) = ui.download.as_ref().filter(|d| !d.finished) {
            let bar_w = client.right - x - metric::INSET - 96;
            label(hdc, x, y + 1, bar_w, &d.line(), ui.fonts.body_bold, t.fg);
            let by = y + 26;
            fill(
                hdc,
                RECT {
                    left: x,
                    top: by,
                    right: x + bar_w,
                    bottom: by + 6,
                },
                t.soft,
            );
            fill(
                hdc,
                RECT {
                    left: x,
                    top: by,
                    right: x + bar_w * d.percent() as i32 / 100,
                    bottom: by + 6,
                },
                t.accent,
            );
            return;
        }

        // The status line is the app's running commentary, right-aligned above
        // the STOP button so it never collides with the endpoint.
        let status = {
            let s = shared().lock().unwrap();
            s.status.clone()
        };

        match &ui.loaded {
            Some(name) => {
                fill(
                    hdc,
                    RECT {
                        left: x,
                        top: y + 8,
                        right: x + 8,
                        bottom: y + 16,
                    },
                    t.green,
                );
                let nx = x + 16;
                label(hdc, nx, y + 1, 320, name, ui.fonts.body_bold, t.fg);
                let url = format!("http://127.0.0.1:{}/v1", ui.port);
                let uw = measure(hdc, &url, ui.fonts.mono) + 8;
                label(hdc, nx, y + 21, uw, &url, ui.fonts.mono, t.accent_text);

                let right = client.right - metric::INSET - 96;
                let left = (nx + 340).min(right - 40);
                let rate = if ui.last_rate > 0.0 {
                    format!("{:.2} tok/s", ui.last_rate)
                } else {
                    "-- tok/s".to_string()
                };
                text(
                    hdc,
                    RECT {
                        left,
                        top: y + 1,
                        right,
                        bottom: y + 19,
                    },
                    &rate,
                    ui.fonts.mono,
                    t.fg,
                    DT_RIGHT | DT_SINGLELINE | DT_END_ELLIPSIS,
                );
                let up = ui
                    .loaded_at
                    .map(|i| human_duration(i.elapsed().as_secs()))
                    .unwrap_or_default();
                text(
                    hdc,
                    RECT {
                        left,
                        top: y + 21,
                        right,
                        bottom: y + 39,
                    },
                    &format!("up {up} · {} tokens · {status}", ui.served),
                    ui.fonts.small,
                    t.fg_tertiary,
                    DT_RIGHT | DT_SINGLELINE | DT_END_ELLIPSIS,
                );
            }
            None => {
                // A hollow dot, and the strip says what to do about it rather
                // than leaving it to be discovered.
                frame(
                    hdc,
                    RECT {
                        left: x,
                        top: y + 8,
                        right: x + 8,
                        bottom: y + 16,
                    },
                    t.stroke_1,
                );
                label(
                    hdc,
                    x + 16,
                    y + 1,
                    420,
                    "no model running",
                    ui.fonts.body_bold,
                    t.fg_secondary,
                );
                label(
                    hdc,
                    x + 16,
                    y + 21,
                    520,
                    "Open MODELS, pick one, and press LOAD.",
                    ui.fonts.small,
                    t.fg_tertiary,
                );
                if !status.is_empty() {
                    text(
                        hdc,
                        RECT {
                            left: x + 560,
                            top: y + 12,
                            right: client.right - metric::INSET - 96,
                            bottom: y + 32,
                        },
                        &status,
                        ui.fonts.small,
                        t.fg_secondary,
                        DT_RIGHT | DT_SINGLELINE | DT_END_ELLIPSIS,
                    );
                }
            }
        }
    }

    /// A page's title block: one display-size line, and one sentence.
    unsafe fn paint_header(hdc: HDC, ui: &Ui, page: RECT, p: Page) {
        let t = &ui.theme;
        let x = page.left + metric::INSET;
        let w = page.right - x - metric::INSET;
        label(hdc, x, page.top + 24, w, p.title(), ui.fonts.display, t.fg);
        label(
            hdc,
            x,
            page.top + 66,
            w,
            p.subtitle(),
            ui.fonts.body,
            t.fg_secondary,
        );
    }

    unsafe fn paint_chat(hdc: HDC, ui: &Ui, page: RECT) {
        paint_header(hdc, ui, page, Page::Chat);
        let x = page.left + metric::INSET;
        label(
            hdc,
            x,
            page.bottom - 24,
            page.right - x - metric::INSET,
            "Ctrl+Enter sends. The conversation is kept until you press CLEAR.",
            ui.fonts.small,
            ui.theme.fg_tertiary,
        );
    }

    /// A model's page: its name, its state, whether it is running, and the
    /// label/value rows underneath.
    type Detail = (String, String, bool, Vec<(String, String)>);

    /// The selected model's own page, as label/value rows.
    fn model_detail(ui: &Ui, sel: Option<usize>) -> Option<Detail> {
        let i = sel?;
        match ui.tab {
            Tab::Installed => {
                let e = ui.entries.get(i)?;
                let running = ui.loaded.as_deref() == Some(e.label.as_str());
                let files = ui.entry_files.get(i).copied().unwrap_or(1);
                let mut rows = vec![
                    (
                        "on disk".into(),
                        format!(
                            "{}{}",
                            models::human_size(e.bytes.unwrap_or(0)),
                            if files > 1 {
                                format!(" across {files} files")
                            } else {
                                String::new()
                            }
                        ),
                    ),
                    (
                        "folder".into(),
                        e.path
                            .parent()
                            .map(|p| p.display().to_string())
                            .unwrap_or_default(),
                    ),
                ];
                if running {
                    rows.push((
                        "endpoint".into(),
                        format!("http://127.0.0.1:{}/v1", ui.port),
                    ));
                    rows.push((
                        "API key".into(),
                        match &ui.cfg.api_key {
                            Some(k) => k.clone(),
                            None => "not required -- any value works".into(),
                        },
                    ));
                    rows.push((
                        "context".into(),
                        ui.cfg
                            .context
                            .map(|c| format!("{c} tokens"))
                            .unwrap_or_else(|| "the model's own limit".into()),
                    ));
                    rows.push((
                        "threads".into(),
                        ui.cfg
                            .threads
                            .map(|c| c.to_string())
                            .unwrap_or_else(|| "measured".into()),
                    ));
                    rows.push((
                        "expert cache".into(),
                        ui.cfg
                            .cache_gib
                            .map(|c| format!("{c} GiB"))
                            .unwrap_or_else(|| "measured".into()),
                    ));
                    rows.push((
                        "uptime".into(),
                        ui.loaded_at
                            .map(|t| human_duration(t.elapsed().as_secs()))
                            .unwrap_or_default(),
                    ));
                    rows.push(("served".into(), format!("{} tokens", ui.served)));
                }
                Some((
                    e.label.clone(),
                    if running {
                        "RUNNING".into()
                    } else {
                        "installed, not running".into()
                    },
                    running,
                    rows,
                ))
            }
            Tab::Available => {
                let o = ui.offers.get(i)?;
                let state = match o.unsupported {
                    // Said first, because it decides whether the rest matters:
                    // "streams" is true and useless if the engine will refuse
                    // the container the moment it is loaded.
                    Some(why) => why,
                    None => match catalog::verdict(o, ui.free_bytes) {
                        catalog::Verdict::Resident => "fits entirely in memory",
                        catalog::Verdict::Streams => "streams from disk on this machine",
                        // **Never "too big".** The whole point of this runner
                        // is models larger than memory; V4-Flash is 144 GB and
                        // works on this machine. What is true is that it will be
                        // slow, and that is what to say.
                        catalog::Verdict::Rereads => {
                            "runs, slowly -- weights are re-read from disk each token"
                        }
                    },
                };
                let rows = vec![
                    ("download".into(), models::human_size(o.bytes)),
                    (
                        "stays resident".into(),
                        format!(
                            "{} -- this is the number that decides",
                            models::human_size(o.always_read)
                        ),
                    ),
                    ("architecture".into(), o.arch.clone()),
                    ("quantisation".into(), o.quant.clone()),
                    (
                        "files".into(),
                        if o.shards > 1 {
                            format!("{} shards", o.shards)
                        } else {
                            "1".into()
                        },
                    ),
                    ("free right now".into(), models::human_size(ui.free_bytes)),
                ];
                Some((o.name.clone(), state.into(), false, rows))
            }
        }
    }

    unsafe fn paint_models(hdc: HDC, ui: &Ui, page: RECT, sel: Option<usize>) {
        paint_header(hdc, ui, page, Page::Models);
        let t = &ui.theme;
        let top = content_top(page);
        let split = models_split(page);

        // A rule under the active tab. A fill would be a second emphasis
        // mechanism for something the eye already reads as a heading.
        let tab_y = top + metric::BUTTON;
        let tx = if ui.tab == Tab::Installed {
            page.left + metric::INSET
        } else {
            page.left + metric::INSET + 104
        };
        fill(
            hdc,
            RECT {
                left: tx,
                top: tab_y,
                right: tx + 96,
                bottom: tab_y + 2,
            },
            t.accent,
        );

        let dx = split + 28;
        let dw = page.right - metric::INSET - dx;
        let mut y = top + 2;

        let Some((name, state, running, rows)) = model_detail(ui, sel) else {
            label(
                hdc,
                dx,
                y + 8,
                dw,
                "Select a model on the left.",
                ui.fonts.body,
                t.fg_tertiary,
            );
            return;
        };

        label(hdc, dx, y, dw, &name, ui.fonts.heading, t.fg);
        y += 28;
        if running {
            fill(
                hdc,
                RECT {
                    left: dx,
                    top: y + 5,
                    right: dx + 7,
                    bottom: y + 12,
                },
                t.green,
            );
            label(hdc, dx + 13, y, dw - 13, &state, ui.fonts.small, t.green);
        } else {
            label(hdc, dx, y, dw, &state, ui.fonts.small, t.fg_tertiary);
        }
        // Below this the buttons sit, positioned by `layout` at the same y.
        y += 24;
        rule(hdc, dx, dx + dw, y, t.stroke_3);
        y += 16 + metric::BUTTON * 2 + 10 + 20;

        // **Every number carries its unit and its meaning**: two aligned
        // columns, no boxes, nothing to click to discover.
        for (k, v) in rows {
            label(hdc, dx, y + 1, 150, &k, ui.fonts.small, t.fg_tertiary);
            label(hdc, dx + 156, y, dw - 156, &v, ui.fonts.mono, t.fg);
            y += 24;
        }
    }

    unsafe fn paint_monitor(hdc: HDC, ui: &Ui, page: RECT) {
        paint_header(hdc, ui, page, Page::Monitor);
        let t = &ui.theme;
        let x = page.left + metric::INSET;
        let w = page.right - x - metric::INSET;
        let mut y = content_top(page);

        let total = ui.total_bytes.max(1);
        let free = ui.free_bytes;
        let used = total.saturating_sub(free);
        let pct = (used.saturating_mul(100) / total) as i32;

        label(hdc, x, y, w, "MEMORY", ui.fonts.small, t.fg_tertiary);
        y += 22;
        label(
            hdc,
            x,
            y,
            w,
            &format!(
                "{} free of {}",
                models::human_size(free),
                models::human_size(total)
            ),
            ui.fonts.heading,
            t.fg,
        );
        y += 32;
        // The only chart in the app, and it earns its place: whether a model
        // fits is the question this page exists to answer.
        let bar_w = w.min(560);
        fill(
            hdc,
            RECT {
                left: x,
                top: y,
                right: x + bar_w,
                bottom: y + 8,
            },
            t.soft,
        );
        fill(
            hdc,
            RECT {
                left: x,
                top: y,
                right: x + bar_w * pct / 100,
                bottom: y + 8,
            },
            if pct > 90 { t.red } else { t.accent },
        );
        y += 16;
        label(
            hdc,
            x,
            y,
            w,
            &format!("{pct}% in use across every process on this machine"),
            ui.fonts.small,
            t.fg_tertiary,
        );
        y += 32;
        rule(hdc, x, x + w, y, t.stroke_3);
        y += 20;

        label(hdc, x, y, w, "GENERATION", ui.fonts.small, t.fg_tertiary);
        y += 24;
        let rows: Vec<(String, String)> = match &ui.loaded {
            Some(name) => vec![
                ("model".into(), name.clone()),
                (
                    "endpoint".into(),
                    format!("http://127.0.0.1:{}/v1", ui.port),
                ),
                (
                    "API key".into(),
                    match &ui.cfg.api_key {
                        Some(k) => k.clone(),
                        None => "not required -- any value works".into(),
                    },
                ),
                (
                    "uptime".into(),
                    ui.loaded_at
                        .map(|i| human_duration(i.elapsed().as_secs()))
                        .unwrap_or_default(),
                ),
                (
                    "last rate".into(),
                    if ui.last_rate > 0.0 {
                        format!("{:.2} tok/s", ui.last_rate)
                    } else {
                        "nothing generated yet".into()
                    },
                ),
                ("tokens served".into(), ui.served.to_string()),
            ],
            None => vec![(
                "model".into(),
                "nothing running -- load one on MODELS".into(),
            )],
        };
        for (k, v) in rows {
            label(hdc, x, y + 1, 160, &k, ui.fonts.small, t.fg_tertiary);
            label(hdc, x + 166, y, w - 166, &v, ui.fonts.mono, t.fg);
            y += 24;
        }
        y += 10;
        rule(hdc, x, x + w, y, t.stroke_3);
        y += 20;

        label(hdc, x, y, w, "ON DISK", ui.fonts.small, t.fg_tertiary);
        y += 24;
        let bytes: u64 = ui.entries.iter().filter_map(|e| e.bytes).sum();
        for (k, v) in [
            (
                "models installed",
                format!(
                    "{}, {} in total",
                    ui.entries.len(),
                    models::human_size(bytes)
                ),
            ),
            ("folder", models::default_dir().display().to_string()),
        ] {
            label(hdc, x, y + 1, 160, k, ui.fonts.small, t.fg_tertiary);
            label(hdc, x + 166, y, w - 166, &v, ui.fonts.mono, t.fg);
            y += 24;
        }

        // **What this page does not know**, said plainly rather than left as an
        // absence. Streamed bytes, expert read rate and cache residency are all
        // measured inside the engine and printed to its log; nothing carries
        // them over the socket yet, so they are not here and are not invented.
        y += 20;
        text(
            hdc,
            RECT {
                left: x,
                top: y,
                right: x + w.min(640),
                bottom: y + 64,
            },
            "Streamed bytes, expert read rate and cache residency are measured inside the \
             engine and printed to its log. They are not on this page because nothing \
             reports them over the socket yet.",
            ui.fonts.small,
            t.fg_tertiary,
            DT_LEFT | DT_WORDBREAK,
        );
    }

    unsafe fn paint_settings(hdc: HDC, ui: &Ui, page: RECT) {
        paint_header(hdc, ui, page, Page::Settings);
        let t = &ui.theme;
        let (left, right, col) = settings_columns(page);

        // Walked exactly as `settings_rows` walks it, so a heading cannot drift
        // away from the boxes underneath it.
        let mut y = [content_top(page), content_top(page)];
        for (gi, group) in nav::GROUPS.iter().enumerate() {
            let c = usize::from(gi % 2 == 1);
            let x = if c == 0 { left } else { right };
            label(hdc, x, y[c] - 2, col, group, ui.fonts.small, t.fg_tertiary);
            rule(hdc, x, x + col, y[c] + 17, t.stroke_3);
            y[c] += 26;
            for f in nav::FIELDS.iter().filter(|f| f.group == *group) {
                label(hdc, x, y[c] - 1, col, f.label, ui.fonts.body_bold, t.fg);
                // **The note describes the current selection, not the field.**
                // A dropdown whose explanation never changes is a dropdown you
                // have to try in order to understand.
                let note = ui
                    .lists
                    .get(&f.id)
                    .and_then(|l| {
                        let sel = unsafe { SendMessageW(sel_of(f.id), CB_GETCURSEL, 0, 0) };
                        (sel >= 0).then(|| l.get(sel as usize)).flatten()
                    })
                    .map(|c| c.note.clone())
                    .unwrap_or_else(|| f.hint.to_string());
                // Below whatever the field puts under itself -- BROWSE, for
                // the models folder -- not on top of it.
                let note_top = y[c] + 18 + metric::CONTROL + 4 + field_extra(f.id);
                text(
                    hdc,
                    RECT {
                        left: x,
                        top: note_top,
                        right: x + col,
                        bottom: note_top + 30,
                    },
                    &note,
                    ui.fonts.small,
                    t.fg_tertiary,
                    DT_LEFT | DT_WORDBREAK | DT_END_ELLIPSIS,
                );
                y[c] += 18 + metric::CONTROL + 34 + field_extra(f.id);
            }
            for f in nav::TOGGLES.iter().filter(|f| f.group == *group) {
                text(
                    hdc,
                    RECT {
                        left: x + 26,
                        top: y[c] + 24,
                        right: x + col,
                        bottom: y[c] + 24 + 28,
                    },
                    f.hint,
                    ui.fonts.small,
                    t.fg_tertiary,
                    DT_LEFT | DT_WORDBREAK | DT_END_ELLIPSIS,
                );
                y[c] += 24 + 30;
            }
            y[c] += 16;
        }

        // Where the file is, because a settings page whose file you cannot find
        // is a settings page you cannot back up or hand to anyone.
        label(
            hdc,
            left,
            page.bottom - 26,
            page.right - left - metric::INSET,
            &format!("Stored in {}", settings::path().display()),
            ui.fonts.small,
            t.fg_tertiary,
        );
    }

    // -- layout --------------------------------------------------------------

    /// Position every control for the current page.
    ///
    /// **Builds the whole move list under the borrow, then applies it with
    /// nothing borrowed.** `MoveWindow` repaints, which asks the parent for
    /// colours, which borrows -- and a `RefCell` double borrow under
    /// `panic = "abort"` is instant, silent process death.
    unsafe fn layout(hwnd: HWND) {
        let mut r = RECT::default();
        GetClientRect(hwnd, &mut r);
        let page = page_rect(r);

        let moves: Vec<(i32, i32, i32, i32, i32)> = UI.with(|u| {
            let b = u.borrow();
            let Some(ui) = b.as_ref() else {
                return Vec::new();
            };
            let mut m = Vec::new();

            for (i, p) in nav::PAGES.iter().enumerate() {
                let q = nav_rect(i);
                m.push((
                    nav::nav_id(*p),
                    q.left,
                    q.top,
                    q.right - q.left,
                    q.bottom - q.top,
                ));
            }
            m.push((
                nav::ID_STRIP_STOP,
                r.right - metric::INSET - 84,
                r.bottom - metric::STRIP + 10,
                84,
                metric::BUTTON,
            ));

            let x = page.left + metric::INSET;
            let w = page.right - x - metric::INSET;
            let top = content_top(page);

            match ui.page {
                Page::Chat => {
                    let composer_h = 88;
                    let out_h = (page.bottom - top - composer_h - 44).max(120);
                    m.push((nav::ID_OUT, x, top, w, out_h));
                    let cy = top + out_h + 14;
                    m.push((nav::ID_IN, x, cy, w - 200, composer_h));
                    m.push((nav::ID_SEND, x + w - 188, cy, 110, metric::BUTTON));
                    m.push((nav::ID_CLEAR, x + w - 70, cy, 70, metric::BUTTON));
                }
                Page::Models => {
                    let split = models_split(page);
                    m.push((nav::ID_TAB_INSTALLED, x, top, 96, metric::BUTTON));
                    m.push((nav::ID_TAB_AVAILABLE, x + 104, top, 96, metric::BUTTON));
                    // RESCAN belongs with the list it refreshes.
                    m.push((nav::ID_REFRESH, split - 84, top, 84, metric::BUTTON));
                    let list_top = top + metric::BUTTON + 14;
                    m.push((
                        nav::ID_LIST,
                        x,
                        list_top,
                        split - x,
                        (page.bottom - list_top - 18).max(80),
                    ));

                    // The model's own actions, under its name and status.
                    let dx = split + 28;
                    let dw = page.right - metric::INSET - dx;
                    let by = top + 2 + 28 + 24 + 16;
                    let bw = ((dw - 20) / 3).clamp(70, 150);
                    m.push((nav::ID_LOAD, dx, by, bw, metric::BUTTON));
                    m.push((nav::ID_UNLOAD, dx + bw + 10, by, bw, metric::BUTTON));
                    m.push((
                        nav::ID_COPY_ENDPOINT,
                        dx + (bw + 10) * 2,
                        by,
                        bw,
                        metric::BUTTON,
                    ));
                    let by2 = by + metric::BUTTON + 10;
                    m.push((nav::ID_GET, dx, by2, bw, metric::BUTTON));
                    m.push((nav::ID_DELETE, dx + bw + 10, by2, bw, metric::BUTTON));
                }
                Page::Monitor => {}
                Page::Settings => {
                    let (_, _, col) = settings_columns(page);
                    let bw = col.min(300);
                    for (id, cx, cy, is_toggle) in settings_rows(page) {
                        // **A combo box's height is the height of its dropped
                        // list, not of the closed box.** Windows sizes the
                        // closed control from its own item height and gives
                        // every remaining pixel to the list, so passing the row
                        // height here left the list nothing to open into:
                        // clicking a dropdown opened a list zero pixels tall,
                        // which looks exactly like a control that does nothing.
                        // That was the whole of "the options never show up".
                        let h = if is_toggle {
                            24
                        } else if choices::for_field(id, ui.machine).is_some() {
                            metric::CONTROL + metric::COMBO_ROW * metric::COMBO_VISIBLE
                        } else {
                            metric::CONTROL
                        };
                        m.push((id, cx, cy, bw, h));
                    }
                    // BROWSE sits directly under the folder it changes, not
                    // beside SAVE: a picker two columns away from the field it
                    // fills is a button nobody connects to anything.
                    if let Some(&(_, fx, fy, _, fh)) =
                        m.iter().find(|(id, ..)| *id == nav::ID_MODELS_DIR)
                    {
                        m.push((nav::ID_BROWSE_MODELS, fx, fy + fh + 6, 120, metric::BUTTON));
                    }
                    let by = page.bottom - 26 - metric::BUTTON - 16;
                    m.push((nav::ID_SAVE, x, by, 110, metric::BUTTON));
                    m.push((nav::ID_RESET, x + 120, by, 110, metric::BUTTON));
                }
            }
            m
        });

        for (id, x, y, w, h) in moves {
            let c = ctl(id);
            if !c.is_null() {
                MoveWindow(c, x, y, w.max(1), h.max(1), 1);
            }
        }
    }

    // -- owner-drawn controls ------------------------------------------------

    /// How a button is weighted.
    ///
    /// Hermes: *"one primary action per page, visually heavier than the rest"*.
    /// The old window had six buttons of identical weight, so the eye had no
    /// idea which one to press.
    #[derive(PartialEq, Clone, Copy)]
    enum Weight {
        /// Filled with the accent. Exactly one per page.
        Primary,
        /// A hairline and a label.
        Secondary,
        /// A hairline and a label, in the destructive colour.
        Destructive,
        /// No box at all -- Hermes' `text` variant, for "Cancel" and "Clear".
        Quiet,
        /// A navigation destination in the rail.
        Nav,
        /// A tab above the model list.
        Tab,
        /// A checkbox and a label.
        Toggle,
    }

    /// **The primary action follows the page**, and on MODELS the tab: loading
    /// what you have, or fetching what you do not.
    fn weight_of(id: i32, page: Page, tab: Tab) -> Weight {
        match id {
            nav::ID_NAV_CHAT | nav::ID_NAV_MODELS | nav::ID_NAV_MONITOR | nav::ID_NAV_SETTINGS => {
                Weight::Nav
            }
            nav::ID_TAB_INSTALLED | nav::ID_TAB_AVAILABLE => Weight::Tab,
            nav::ID_AUTO | nav::ID_FORCE => Weight::Toggle,
            nav::ID_DELETE => Weight::Destructive,
            nav::ID_CLEAR | nav::ID_RESET | nav::ID_REFRESH => Weight::Quiet,
            nav::ID_SEND if page == Page::Chat => Weight::Primary,
            nav::ID_SAVE if page == Page::Settings => Weight::Primary,
            nav::ID_LOAD if page == Page::Models && tab == Tab::Installed => Weight::Primary,
            nav::ID_GET if page == Page::Models && tab == Tab::Available => Weight::Primary,
            _ => Weight::Secondary,
        }
    }

    /// Paint one owner-drawn control.
    ///
    /// Buttons and the list selection are the two places Windows insists on its
    /// own colours -- a themed push button ignores `WM_CTLCOLORBTN` entirely,
    /// and the selection bar is the system highlight. Both are drawn here.
    unsafe fn draw_item(di: &DRAWITEMSTRUCT) {
        let selected = di.itemState & ODS_SELECTED != 0;
        let disabled = di.itemState & ODS_DISABLED != 0;
        let focused = di.itemState & ODS_FOCUS != 0;

        // Everything needed comes out of the borrow first; the reads from
        // Windows below are all outside it.
        let snapshot = UI.with(|u| {
            let b = u.borrow();
            b.as_ref().map(|ui| {
                (
                    ui.theme,
                    ui.page,
                    ui.tab,
                    ui.fonts.body,
                    ui.fonts.body_bold,
                    ui.cfg.auto,
                    ui.cfg.force,
                    ui.loaded.clone(),
                )
            })
        });
        let Some((t, page, tab, font, font_bold, auto, force, loaded)) = snapshot else {
            return;
        };

        if di.CtlType == ODT_LISTBOX {
            draw_list_row(di, &t, font, font_bold, selected, loaded.as_deref());
            return;
        }
        if di.CtlType == ODT_COMBOBOX {
            draw_combo(di, &t, font, font_bold, selected);
            return;
        }

        let id = di.CtlID as i32;
        let text_s = control_text(di.hwndItem);
        let r = di.rcItem;

        match weight_of(id, page, tab) {
            Weight::Nav => {
                let active = nav::page_of_nav(id) == Some(page);
                let bg = if active {
                    t.accent_soft
                } else if selected {
                    t.soft
                } else {
                    t.chrome
                };
                fill(di.hDC, r, bg);
                if active {
                    // A rule down the left edge as well as the wash: the marker
                    // has to survive on a ground that is already tinted.
                    fill(
                        di.hDC,
                        RECT {
                            left: r.left,
                            top: r.top + 4,
                            right: r.left + 3,
                            bottom: r.bottom - 4,
                        },
                        t.accent,
                    );
                }
                text(
                    di.hDC,
                    RECT {
                        left: r.left + 16,
                        ..r
                    },
                    &text_s,
                    if active { font_bold } else { font },
                    if active {
                        t.accent_text
                    } else {
                        t.fg_secondary
                    },
                    DT_LEFT | DT_SINGLELINE | DT_VCENTER,
                );
            }
            Weight::Tab => {
                let active = (id == nav::ID_TAB_INSTALLED) == (tab == Tab::Installed);
                fill(di.hDC, r, t.bg);
                text(
                    di.hDC,
                    r,
                    &text_s,
                    if active { font_bold } else { font },
                    if active { t.fg } else { t.fg_tertiary },
                    DT_CENTER | DT_SINGLELINE | DT_VCENTER,
                );
            }
            Weight::Toggle => {
                fill(di.hDC, r, t.bg);
                let on = if id == nav::ID_AUTO { auto } else { force };
                let b = RECT {
                    left: r.left,
                    top: r.top + 3,
                    right: r.left + 16,
                    bottom: r.top + 19,
                };
                if on {
                    fill(di.hDC, b, t.accent);
                    // A tick drawn as two strokes: a glyph would depend on a
                    // font having one at this size, and Segoe UI's is centred
                    // for a much larger box.
                    let pen = CreatePen(0, 2, t.on_accent);
                    let old = SelectObject(di.hDC, pen);
                    MoveToEx(di.hDC, b.left + 4, b.top + 8, std::ptr::null_mut());
                    LineTo(di.hDC, b.left + 7, b.top + 11);
                    LineTo(di.hDC, b.left + 12, b.top + 5);
                    SelectObject(di.hDC, old);
                    DeleteObject(pen);
                } else {
                    frame(di.hDC, b, t.stroke_1);
                }
                text(
                    di.hDC,
                    RECT {
                        left: r.left + 26,
                        ..r
                    },
                    &text_s,
                    font,
                    t.fg,
                    DT_LEFT | DT_SINGLELINE | DT_VCENTER,
                );
                if focused {
                    frame(di.hDC, inset(b, -3), t.stroke_1);
                }
            }
            Weight::Primary => {
                let bg = if disabled {
                    t.soft
                } else if selected {
                    // Pressed: the same blue, taken down. There is no second
                    // accent to reach for, and inverting would read as disabled.
                    mix(t.accent, t.fg, 22)
                } else {
                    t.accent
                };
                fill(di.hDC, r, bg);
                text(
                    di.hDC,
                    r,
                    &text_s,
                    font_bold,
                    if disabled { t.fg_tertiary } else { t.on_accent },
                    DT_CENTER | DT_SINGLELINE | DT_VCENTER,
                );
                if focused && !disabled {
                    frame(di.hDC, inset(r, 3), t.on_accent);
                }
            }
            w @ (Weight::Secondary | Weight::Destructive) => {
                let danger = w == Weight::Destructive;
                fill(di.hDC, r, if selected { t.soft_active } else { t.bg });
                // The hairline is how a button says it can be pressed. Omitted
                // when disabled: removing the rule is the quiet signal, and
                // inverting the whole control was the loud one -- which used to
                // land on precisely the controls that do nothing.
                if !disabled {
                    frame(di.hDC, r, if danger { t.red } else { t.stroke_1 });
                }
                text(
                    di.hDC,
                    r,
                    &text_s,
                    font,
                    if disabled {
                        t.fg_tertiary
                    } else if danger {
                        t.red
                    } else {
                        t.fg
                    },
                    DT_CENTER | DT_SINGLELINE | DT_VCENTER,
                );
                if focused && !disabled {
                    frame(di.hDC, inset(r, 3), t.stroke_2);
                }
            }
            Weight::Quiet => {
                fill(di.hDC, r, if selected { t.soft } else { t.bg });
                text(
                    di.hDC,
                    r,
                    &text_s,
                    font,
                    if disabled {
                        t.fg_tertiary
                    } else {
                        t.fg_secondary
                    },
                    DT_CENTER | DT_SINGLELINE | DT_VCENTER,
                );
            }
        }
    }

    /// A dropdown: its closed face, and its rows when open.
    ///
    /// Owner-drawn for the same reason the buttons are -- a themed combo ignores
    /// `WM_CTLCOLOR*` and comes up in the system's greys. `ODS_COMBOBOXEDIT`
    /// distinguishes the closed control from a row in the open list, which are
    /// drawn differently: the face carries a chevron, the rows do not.
    unsafe fn draw_combo(
        di: &DRAWITEMSTRUCT,
        t: &Theme,
        font: HFONT,
        font_bold: HFONT,
        selected: bool,
    ) {
        let face = di.itemState & ODS_COMBOBOXEDIT != 0;
        let r = di.rcItem;
        // The open list sits on the elevated fill so it reads as floating; the
        // closed face sits on the page.
        let ground = if face {
            t.bg
        } else if selected {
            t.accent_soft
        } else {
            t.soft
        };
        fill(di.hDC, r, ground);

        if face {
            frame(di.hDC, r, t.stroke_1);
            // **No chevron is drawn here.** `CBS_OWNERDRAWFIXED` hands over the
            // *items*, not the drop-down button, which Windows keeps painting
            // itself -- so drawing one produced two, side by side.
        } else if selected {
            fill(
                di.hDC,
                RECT {
                    right: r.left + 3,
                    ..r
                },
                t.accent,
            );
        }

        if di.itemID == u32::MAX {
            return;
        }
        // The text comes from our own cached list rather than `CB_GETLBTEXT`:
        // the label is already in hand, and asking the control for it during
        // its own paint is a message round trip for nothing.
        let label = UI.with(|u| {
            let b = u.borrow();
            b.as_ref()
                .and_then(|ui| ui.lists.get(&(di.CtlID as i32)))
                .and_then(|l| l.get(di.itemID as usize))
                .map(|c| c.label.clone())
        });
        let Some(label) = label else { return };
        text(
            di.hDC,
            RECT {
                left: r.left + 10,
                right: r.right - if face { 26 } else { 8 },
                ..r
            },
            &label,
            if face { font_bold } else { font },
            t.fg,
            DT_LEFT | DT_SINGLELINE | DT_VCENTER | DT_END_ELLIPSIS,
        );
    }

    /// One row of the model list.
    unsafe fn draw_list_row(
        di: &DRAWITEMSTRUCT,
        t: &Theme,
        font: HFONT,
        font_bold: HFONT,
        selected: bool,
        loaded: Option<&str>,
    ) {
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
        let s = String::from_utf16_lossy(&buf[..n as usize]);

        fill(
            di.hDC,
            di.rcItem,
            if selected { t.soft_active } else { t.bg },
        );
        if selected {
            fill(
                di.hDC,
                RECT {
                    right: di.rcItem.left + 3,
                    ..di.rcItem
                },
                t.accent,
            );
        }
        // A running model is marked in the list as well as on the strip, so
        // "which one is up" is answered wherever you happen to be looking.
        let running = loaded.is_some_and(|l| s.starts_with(l));
        let mut x = di.rcItem.left + 12;
        if running {
            fill(
                di.hDC,
                RECT {
                    left: x,
                    top: di.rcItem.top + 10,
                    right: x + 7,
                    bottom: di.rcItem.top + 17,
                },
                t.green,
            );
            x += 13;
        }
        // **Columns, drawn right to left.** Everything after the first is a
        // measurement -- the size, and "(unfinished)" -- and each is placed at
        // its own right edge so the name keeps every pixel that is left. Drawn
        // as one string it was the *name* that lost its tail to the ellipsis,
        // and a name without its quantisation does not identify a model.
        let mut parts = s.split(models::COLUMN_SEP);
        let name = parts.next().unwrap_or("");
        let extras: Vec<&str> = parts.collect();
        let mut right = di.rcItem.right - 12;
        for extra in extras.iter().rev() {
            let w = text_width(di.hDC, extra, font) + 18;
            text(
                di.hDC,
                RECT {
                    left: right - w,
                    right,
                    ..di.rcItem
                },
                extra,
                font,
                t.fg_tertiary,
                DT_RIGHT | DT_SINGLELINE | DT_VCENTER,
            );
            right -= w;
        }
        text(
            di.hDC,
            RECT {
                left: x,
                right: right.max(x + 40),
                ..di.rcItem
            },
            name,
            if running { font_bold } else { font },
            t.fg,
            DT_LEFT | DT_SINGLELINE | DT_VCENTER | DT_END_ELLIPSIS,
        );
    }

    /// How wide a string draws in `font`, for right-aligning a column.
    unsafe fn text_width(hdc: HDC, s: &str, font: HFONT) -> i32 {
        let old = SelectObject(hdc, font as HGDIOBJ);
        let w: Vec<u16> = s.encode_utf16().collect();
        let mut sz = SIZE { cx: 0, cy: 0 };
        if !w.is_empty() {
            GetTextExtentPoint32W(hdc, w.as_ptr(), w.len() as i32, &mut sz);
        }
        SelectObject(hdc, old);
        sz.cx
    }

    /// Put the embedded icon on the window itself.
    ///
    /// The resource compiled into the executable is what Explorer shows for the
    /// *file*. The title bar, the taskbar button and the Alt-Tab strip read the
    /// *window's* icon, which is unset until something sends `WM_SETICON`.
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

    // ---- updating in place -------------------------------------------------
    //
    // Atur: *"users can get the most updated release when they connect to the
    // internet from the app -- an updating flow, not every time go and download
    // a new setup"*.
    //
    // The decisions -- is this newer, which file does this platform need -- are
    // in `chaos_app::update`, where they are tested. What is here is the two
    // things that need a window: a `curl` call on a worker thread, and the
    // sequencing of the handover to the installer.

    /// Ask GitHub what the newest release is, without flashing a console.
    ///
    /// The arguments are `update::feed_curl_args()`, the same list
    /// `chaos-run --update` uses -- the missing `User-Agent` that the API
    /// rejects outright is exactly the kind of thing that gets fixed in one
    /// copy. What is here and not shared is `CREATE_NO_WINDOW`, a Windows-only
    /// extension to `Command`: without it a console window blinks open on the
    /// user's screen every time the app checks.
    fn fetch_latest_json() -> Result<String, String> {
        use std::os::windows::process::CommandExt;
        let mut cmd = Command::new("curl");
        cmd.args(update::feed_curl_args());
        cmd.creation_flags(CREATE_NO_WINDOW);
        match cmd.output() {
            Ok(o) if o.status.success() => Ok(String::from_utf8_lossy(&o.stdout).into_owned()),
            Ok(o) => {
                let why = String::from_utf8_lossy(&o.stderr);
                let first = why.lines().next().unwrap_or("").trim().to_string();
                Err(if first.is_empty() {
                    format!("curl exited {}", o.status)
                } else {
                    first
                })
            }
            Err(e) => Err(format!("curl could not be run ({e})")),
        }
    }

    /// Check in the background, and leave the answer for `drain` to show.
    ///
    /// `announce` is what separates the two callers. The one at startup is
    /// quiet unless there is news -- an app that opens a dialog to say nothing
    /// has changed is an app people stop launching. The menu item always
    /// answers, because a check that says nothing is indistinguishable from a
    /// broken one.
    fn check_for_updates(announce: bool) {
        if announce {
            set_status("checking for a newer Chaos...");
        }
        std::thread::spawn(move || {
            let outcome = match fetch_latest_json() {
                Ok(json) => update::decide(update::parse_latest(&json), update::running()),
                Err(why) => update::Outcome::Failed(why),
            };
            let news = announce || matches!(outcome, update::Outcome::Available { .. });
            {
                let mut sh = shared().lock().unwrap();
                sh.update = Some(outcome);
                sh.update_news = news;
                sh.update_asked = announce;
            }
            notify();
        });
    }

    /// Show what a finished check found, once.
    ///
    /// Runs on the UI thread out of `drain`, which is the only place a dialog
    /// can be put up at all: `MessageBoxW` owned by a window belonging to
    /// another thread does nothing here, as the connection test found out.
    fn show_update_outcome() {
        let (outcome, asked) = {
            let mut sh = shared().lock().unwrap();
            if !sh.update_news {
                return;
            }
            sh.update_news = false;
            (sh.update.clone(), sh.update_asked)
        };
        let Some(outcome) = outcome else { return };
        set_status(&outcome.line());
        match &outcome {
            update::Outcome::Available { version, .. } => {
                let msg = format!(
                    "Chaos {} is available. You are running {}.\r\n\r\n\
                     Download it and start the installer now?\r\n\r\n\
                     Chaos will close so the installer can replace its files. \
                     Your models are not touched.",
                    version.text(),
                    update::running().text()
                );
                let yes = unsafe {
                    MessageBoxW(
                        main_hwnd(),
                        wide(&msg).as_ptr(),
                        wide("A newer Chaos").as_ptr(),
                        MB_YESNO | MB_ICONINFORMATION,
                    )
                } == IDYES;
                if yes {
                    install_update();
                }
            }
            // Only the menu item's caller waited for these, so only it hears
            // them as a dialog; the status line has them either way.
            _ if asked => unsafe {
                MessageBoxW(
                    main_hwnd(),
                    wide(&outcome.line()).as_ptr(),
                    wide("Check for updates").as_ptr(),
                    MB_OK | MB_ICONINFORMATION,
                );
            },
            _ => {}
        }
    }

    /// Fetch the installer for the release the last check found, and hand over.
    ///
    /// **The app has to be gone before the installer writes.** `chaos-setup`
    /// overwrites `chaos-app.exe` in place, and Windows keeps a running
    /// executable's file open -- so a silent install from inside the running app
    /// would fail on exactly the binary doing the asking. The installer is
    /// therefore started with its window and this process closes immediately; by
    /// the time anyone presses INSTALL there is nothing holding the file.
    fn install_update() {
        let outcome = shared().lock().unwrap().update.clone();
        let (version, url) = match outcome {
            Some(update::Outcome::Available { version, url }) => (version, url),
            Some(update::Outcome::UpToDate(v)) => {
                set_status(&format!("Chaos {} is already the newest release", v.text()));
                return;
            }
            // Nothing has been checked yet, so check rather than refuse.
            _ => {
                check_for_updates(true);
                return;
            }
        };
        let name = update::asset_for_platform(&version);
        let dest = std::env::temp_dir().join(&name);
        set_status(&format!("downloading Chaos {}...", version.text()));
        std::thread::spawn(move || {
            use std::os::windows::process::CommandExt;
            let mut cmd = Command::new("curl");
            cmd.args(["-L", "--fail", "-sS", "--retry", "3"])
                .arg("-o")
                .arg(&dest)
                .arg(&url);
            cmd.creation_flags(CREATE_NO_WINDOW);
            let ok = matches!(cmd.status(), Ok(st) if st.success());
            // **Exit zero is not a file.** The same trap `chaos-pull` documents:
            // a redirect to an error page saves perfectly and runs not at all.
            // No installer this project has ever built is under a megabyte.
            let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            if !ok || bytes < (1 << 20) {
                let mut sh = shared().lock().unwrap();
                sh.report = Some((
                    format!(
                        "The update could not be downloaded.\n\n{url}\n\n\
                         Download it in a browser and run it -- your models and \
                         settings are kept."
                    ),
                    false,
                ));
                drop(sh);
                notify();
                return;
            }
            let started = Command::new(&dest).spawn().is_ok();
            let mut sh = shared().lock().unwrap();
            sh.update_quit = started;
            if !started {
                sh.report = Some((
                    format!(
                        "The update downloaded but would not start. Run it by hand:\n\n{}",
                        dest.display()
                    ),
                    false,
                ));
            }
            drop(sh);
            notify();
        });
    }

    fn about() {
        let msg = format!(
            "Chaos v{}\n\nA runner for models larger than memory: the always-read \
             weights stay resident, routed experts stream from disk per token.\n\n\
             {RELEASES_URL}",
            env!("CARGO_PKG_VERSION")
        );
        unsafe {
            MessageBoxW(
                main_hwnd(),
                wide(&msg).as_ptr(),
                wide("About Chaos").as_ptr(),
                MB_OK | MB_ICONINFORMATION,
            );
        }
    }

    fn open_crash_log() {
        let p = std::env::temp_dir().join("chaos-app-crash.log");
        if p.exists() {
            shell_open(&p.display().to_string());
        } else {
            set_status("no crash log -- Chaos has not crashed on this machine");
        }
    }

    fn set_tab(tab: Tab) {
        UI.with(|u| {
            if let Some(ui) = u.borrow_mut().as_mut() {
                ui.tab = tab;
            }
        });
        rescan();
    }

    unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        match msg {
            WM_SIZE => {
                layout(hwnd);
                InvalidateRect(hwnd, std::ptr::null(), 1);
                0
            }
            // Below this the rail plus a page has nowhere to put anything.
            WM_GETMINMAXINFO => {
                let mm = &mut *(lp as *mut MINMAXINFO);
                mm.ptMinTrackSize.x = MIN_W;
                mm.ptMinTrackSize.y = MIN_H;
                0
            }
            // Answered here so Windows never paints a ground we are about to
            // paint over: that flash of the wrong colour is the whole of it.
            WM_ERASEBKGND => 1,
            WM_PAINT => {
                paint(hwnd);
                0
            }
            WM_TIMER => {
                // Uptime and free memory move on their own; nothing else here
                // would ever ask for them to be redrawn.
                let free = free_memory_bytes();
                // **A child that has exited is not a running model.** The
                // window used to keep the green dot and the endpoint up after
                // `chaos-serve` died, so the next message failed with a
                // connection error and no explanation.
                let died = UI.with(|u| {
                    let mut b = u.borrow_mut();
                    let ui = b.as_mut()?;
                    let gone = match ui.server.as_mut() {
                        Some(c) => matches!(c.try_wait(), Ok(Some(_))),
                        None => false,
                    };
                    if !gone {
                        return None;
                    }
                    let name = ui.loaded.clone();
                    ui.server = None;
                    ui.loaded = None;
                    ui.loaded_at = None;
                    Some(name.unwrap_or_default())
                });
                if let Some(name) = died {
                    sync_enabled();
                    set_status(&format!(
                        "{name} stopped on its own -- the engine could not keep it running"
                    ));
                }
                // A download is another process writing files, so the only way
                // to know how far along it is, is to look at them.
                let ended = {
                    let mut sh = shared().lock().unwrap();
                    std::mem::take(&mut sh.download_done)
                };
                let mut rescan_after = false;
                UI.with(|u| {
                    if let Some(ui) = u.borrow_mut().as_mut() {
                        ui.free_bytes = free;
                        if let Some(d) = ui.download.as_mut() {
                            d.done_bytes = chaos_app::download::bytes_on_disk(&d.files);
                            d.elapsed += f64::from(TICK_MS) / 1000.0;
                            if ended {
                                d.finished = true;
                                rescan_after = true;
                            }
                        }
                    }
                });
                if rescan_after {
                    // The finished container is now an installed model.
                    rescan();
                }
                repaint();
                0
            }
            WM_INITMENUPOPUP => {
                sync_menu(hwnd);
                DefWindowProcW(hwnd, msg, wp, lp)
            }
            // Every control repainted to the palette. Without these the boxes
            // come up in the system's greys, which is the whole design gone.
            WM_CTLCOLOREDIT | WM_CTLCOLORLISTBOX | WM_CTLCOLORSTATIC | WM_CTLCOLORBTN => {
                UI.with(|u| {
                    let b = u.borrow();
                    let Some(ui) = b.as_ref() else { return 0 };
                    SetTextColor(wp as HDC, ui.theme.fg);
                    SetBkColor(wp as HDC, ui.theme.bg);
                    ui.brushes.bg as LRESULT
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
            // Windows asks how tall a row is before it draws one, and the
            // default is the system font's height -- which clips a 15px label.
            WM_MEASUREITEM => {
                let mi = &mut *(lp as *mut MEASUREITEMSTRUCT);
                mi.itemHeight = 26;
                1
            }
            WM_COMMAND => {
                let id = (wp & 0xFFFF) as i32;
                let code = ((wp >> 16) & 0xFFFF) as u16;
                // A menu pick and a rail button end in the same call, which is
                // the rule about one action having exactly one home.
                if let Some(p) = nav::page_of_menu(id).or_else(|| nav::page_of_nav(id)) {
                    show_page(p);
                    return 0;
                }
                match id {
                    nav::IDM_EXIT => {
                        DestroyWindow(hwnd);
                        return 0;
                    }
                    nav::IDM_THEME_LIGHT => {
                        set_mode(Mode::Light);
                        return 0;
                    }
                    nav::IDM_THEME_DARK => {
                        set_mode(Mode::Dark);
                        return 0;
                    }
                    nav::IDM_RESCAN => {
                        rescan();
                        return 0;
                    }
                    nav::IDM_OPEN_MODELS_DIR => {
                        let d = models::default_dir();
                        let _ = std::fs::create_dir_all(&d);
                        shell_open(&d.display().to_string());
                        return 0;
                    }
                    nav::IDM_LOAD => {
                        load_model();
                        return 0;
                    }
                    nav::IDM_STOP => {
                        unload_model();
                        return 0;
                    }
                    nav::IDM_DOWNLOAD => {
                        download_selected();
                        return 0;
                    }
                    nav::IDM_DELETE => {
                        delete_selected();
                        return 0;
                    }
                    nav::IDM_COPY_ENDPOINT => {
                        copy_endpoint();
                        return 0;
                    }
                    nav::IDM_API_KEY => {
                        toggle_api_key();
                        return 0;
                    }
                    nav::IDM_TEST_CONNECTION => {
                        test_connection();
                        return 0;
                    }
                    nav::IDM_MANUAL => {
                        shell_open(MANUAL_URL);
                        return 0;
                    }
                    nav::IDM_RELEASES => {
                        shell_open(RELEASES_URL);
                        return 0;
                    }
                    nav::IDM_CRASH_LOG => {
                        open_crash_log();
                        return 0;
                    }
                    nav::IDM_ABOUT => {
                        about();
                        return 0;
                    }
                    nav::IDM_CHECK_UPDATE => {
                        check_for_updates(true);
                        return 0;
                    }
                    nav::IDM_INSTALL_UPDATE => {
                        install_update();
                        return 0;
                    }
                    _ => {}
                }
                match (id, code) {
                    (nav::ID_LOAD, BN_CLICKED) => load_model(),
                    (nav::ID_UNLOAD, BN_CLICKED) | (nav::ID_STRIP_STOP, BN_CLICKED) => {
                        unload_model()
                    }
                    // `BN_CLICKED` is zero, which is also the code an
                    // accelerator arrives with -- so Ctrl+Enter and the button
                    // land on this one arm rather than needing two.
                    (nav::ID_SEND, BN_CLICKED) => send_prompt(),
                    (nav::ID_CLEAR, BN_CLICKED) => clear_chat(),
                    (nav::ID_REFRESH, BN_CLICKED) => rescan(),
                    (nav::ID_GET, BN_CLICKED) => download_selected(),
                    (nav::ID_DELETE, BN_CLICKED) => delete_selected(),
                    (nav::ID_COPY_ENDPOINT, BN_CLICKED) => copy_endpoint(),
                    (nav::ID_TAB_INSTALLED, BN_CLICKED) => set_tab(Tab::Installed),
                    (nav::ID_TAB_AVAILABLE, BN_CLICKED) => set_tab(Tab::Available),
                    (nav::ID_SAVE, BN_CLICKED) => save_settings(),
                    (nav::ID_BROWSE_MODELS, BN_CLICKED) => browse_models_dir(hwnd),
                    (nav::ID_RESET, BN_CLICKED) => reset_settings(),
                    (nav::ID_AUTO, BN_CLICKED) | (nav::ID_FORCE, BN_CLICKED) => toggle(id),
                    // Selecting a different model redraws its page beside the
                    // list, which is the whole point of a page per model --
                    // **and re-decides which buttons are live**, because what
                    // can be done to a model now depends on the model. An
                    // unfinished download offers DOWNLOAD and refuses LOAD; a
                    // whole one is the other way round. Repainting alone left
                    // whichever answer the *first* row happened to give.
                    (nav::ID_LIST, LBN_SELCHANGE) => {
                        sync_enabled();
                        repaint();
                    }
                    // A settings dropdown changed: the sentence under it
                    // describes the *selected* option, so it has to be redrawn.
                    (_, CBN_SELCHANGE) => repaint(),
                    _ => {}
                }
                0
            }
            WM_CLOSE => {
                stop_server();
                DestroyWindow(hwnd);
                0
            }
            WM_DESTROY => {
                // **Closing the window must stop the engine.** The model runs
                // in a child `chaos-serve`, and until this was here, closing
                // Chaos left that child alive holding every resident byte --
                // 7 GiB for V4-Flash -- with no window to stop it from. The
                // taskbar close became a memory leak you had to find in Task
                // Manager.
                KillTimer(hwnd, TIMER_ID);
                stop_server();
                PostQuitMessage(0);
                0
            }
            _ => DefWindowProcW(hwnd, msg, wp, lp),
        }
    }
}
