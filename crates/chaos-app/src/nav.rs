//! Where everything lives.
//!
//! The old window put a model list, a download catalogue, four actions and
//! three settings into one 380px column, and Atur's verdict was the correct
//! one: *"why is all click in one slot"*. The answer is that nothing had been
//! given a home, so everything ended up in the same one.
//!
//! This module is that decision, written down as data: four destinations, and
//! for each control, the single page it belongs to. Hermes' `DESIGN.md` calls
//! these *durable destinations* -- "do not hide a distinct product noun inside
//! an unrelated page" -- and that is the rule the tests at the bottom enforce.
//!
//! No Win32 here either. A page is an enum and a control is a number, so the
//! structure of the app can be checked on a machine with no window server.

/// The four destinations.
///
/// Ordered as they appear in the rail, which is also the order of their
/// accelerators (`Ctrl+1` .. `Ctrl+4`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Page {
    /// The conversation. Hermes: *"chat is the home surface"*, and it is the
    /// reason anyone opens this app.
    Chat,
    /// What is installed, what can be fetched, and one page per model.
    Models,
    /// What the machine is doing while a model runs.
    Monitor,
    /// Everything the settings file holds -- which is nine fields, of which the
    /// old window showed three.
    Settings,
}

pub const PAGES: [Page; 4] = [Page::Chat, Page::Models, Page::Monitor, Page::Settings];

impl Page {
    /// The label in the navigation rail.
    pub fn label(self) -> &'static str {
        match self {
            Page::Chat => "CHAT",
            Page::Models => "MODELS",
            Page::Monitor => "MONITOR",
            Page::Settings => "SETTINGS",
        }
    }

    /// The page's own title, at display size, once at the top.
    pub fn title(self) -> &'static str {
        match self {
            Page::Chat => "Chat",
            Page::Models => "Models",
            Page::Monitor => "Monitor",
            Page::Settings => "Settings",
        }
    }

    /// One line under the title saying what the page is for.
    ///
    /// Hermes' rule: *"a control should say exactly what happens when it's
    /// used"*. A page owes the same sentence.
    pub fn subtitle(self) -> &'static str {
        match self {
            Page::Chat => "Talk to the running model, or point a coding agent at its endpoint.",
            Page::Models => "What is on this machine, and what Chaos can fetch.",
            Page::Monitor => "What the machine is doing while a model runs.",
            Page::Settings => "Every setting Chaos keeps. Empty means measured.",
        }
    }

    /// `Ctrl+<n>` reaches this page.
    pub fn accel(self) -> u8 {
        match self {
            Page::Chat => b'1',
            Page::Models => b'2',
            Page::Monitor => b'3',
            Page::Settings => b'4',
        }
    }

    pub fn index(self) -> usize {
        PAGES.iter().position(|&p| p == self).unwrap_or(0)
    }
}

// -- control identifiers -----------------------------------------------------
//
// One block, numbered by page, so a glance at an id says where it lives.

// Chat: 100
pub const ID_OUT: i32 = 101;
pub const ID_IN: i32 = 102;
pub const ID_SEND: i32 = 103;
pub const ID_CLEAR: i32 = 104;

// Models: 200
pub const ID_TAB_INSTALLED: i32 = 201;
pub const ID_TAB_AVAILABLE: i32 = 202;
pub const ID_LIST: i32 = 203;
pub const ID_LOAD: i32 = 204;
pub const ID_UNLOAD: i32 = 205;
pub const ID_GET: i32 = 206;
pub const ID_DELETE: i32 = 207;
pub const ID_REFRESH: i32 = 208;
pub const ID_COPY_ENDPOINT: i32 = 209;

// Settings: 300
pub const ID_CACHE: i32 = 301;
pub const ID_THREADS: i32 = 302;
pub const ID_THREADS_BATCH: i32 = 303;
pub const ID_PORT: i32 = 304;
pub const ID_CONTEXT: i32 = 305;
pub const ID_NGL: i32 = 306;
pub const ID_MODELS_DIR: i32 = 307;
pub const ID_AUTO: i32 = 308;
pub const ID_FORCE: i32 = 309;
pub const ID_SAVE: i32 = 310;
pub const ID_RESET: i32 = 311;
/// Pick the models folder with a dialog instead of typing a path.
pub const ID_BROWSE_MODELS: i32 = 312;

// The shell: 400. Present on every page.
pub const ID_NAV_CHAT: i32 = 401;
pub const ID_NAV_MODELS: i32 = 402;
pub const ID_NAV_MONITOR: i32 = 403;
pub const ID_NAV_SETTINGS: i32 = 404;
pub const ID_STRIP_STOP: i32 = 405;

// Menu commands: 500. A separate range so `WM_COMMAND` can tell a menu pick
// from a button press without consulting the high word.
pub const IDM_RESCAN: i32 = 501;
pub const IDM_OPEN_MODELS_DIR: i32 = 502;
pub const IDM_EXIT: i32 = 503;
pub const IDM_LOAD: i32 = 510;
pub const IDM_STOP: i32 = 511;
pub const IDM_DOWNLOAD: i32 = 512;
pub const IDM_DELETE: i32 = 513;
pub const IDM_COPY_ENDPOINT: i32 = 514;
pub const IDM_API_KEY: i32 = 515;
pub const IDM_TEST_CONNECTION: i32 = 516;
pub const IDM_PAGE_CHAT: i32 = 520;
pub const IDM_PAGE_MODELS: i32 = 521;
pub const IDM_PAGE_MONITOR: i32 = 522;
pub const IDM_PAGE_SETTINGS: i32 = 523;
pub const IDM_THEME_LIGHT: i32 = 524;
pub const IDM_THEME_DARK: i32 = 525;
pub const IDM_MANUAL: i32 = 530;
pub const IDM_RELEASES: i32 = 531;
pub const IDM_CRASH_LOG: i32 = 532;
pub const IDM_ABOUT: i32 = 533;
pub const IDM_CHECK_UPDATE: i32 = 534;
pub const IDM_INSTALL_UPDATE: i32 = 535;

// The notification-area menu. Not on the menu bar, so `every_menu_command_is_handled`
// would look for them there -- they are 6xx to keep that distinction visible.
/// Bring the window back from the notification area.
pub const IDM_TRAY_OPEN: i32 = 601;
/// Quit for real, as opposed to closing the window.
pub const IDM_TRAY_EXIT: i32 = 602;

/// The navigation button for a page.
pub fn nav_id(p: Page) -> i32 {
    match p {
        Page::Chat => ID_NAV_CHAT,
        Page::Models => ID_NAV_MODELS,
        Page::Monitor => ID_NAV_MONITOR,
        Page::Settings => ID_NAV_SETTINGS,
    }
}

/// The page a `View` menu command selects, if it is one.
pub fn page_of_menu(id: i32) -> Option<Page> {
    match id {
        IDM_PAGE_CHAT => Some(Page::Chat),
        IDM_PAGE_MODELS => Some(Page::Models),
        IDM_PAGE_MONITOR => Some(Page::Monitor),
        IDM_PAGE_SETTINGS => Some(Page::Settings),
        _ => None,
    }
}

/// The page a navigation button selects, if it is one.
pub fn page_of_nav(id: i32) -> Option<Page> {
    PAGES.iter().copied().find(|&p| nav_id(p) == id)
}

/// The controls belonging to each page.
///
/// **A control appears on exactly one page.** Everything else is either shell
/// chrome (the rail, the strip) or painted rather than made into a window.
/// This is the list `show_page` walks to hide and reveal, so a control missing
/// from it is a control that never appears.
pub fn controls(p: Page) -> &'static [i32] {
    match p {
        Page::Chat => &[ID_OUT, ID_IN, ID_SEND, ID_CLEAR],
        Page::Models => &[
            ID_TAB_INSTALLED,
            ID_TAB_AVAILABLE,
            ID_LIST,
            ID_LOAD,
            ID_UNLOAD,
            ID_GET,
            ID_DELETE,
            ID_REFRESH,
            ID_COPY_ENDPOINT,
        ],
        // Painted entirely. Every number on it is read from the machine each
        // tick, so a static control would be a second place to keep in step.
        Page::Monitor => &[],
        Page::Settings => &[
            ID_CACHE,
            ID_THREADS,
            ID_THREADS_BATCH,
            ID_PORT,
            ID_CONTEXT,
            ID_NGL,
            ID_MODELS_DIR,
            ID_AUTO,
            ID_FORCE,
            ID_SAVE,
            ID_RESET,
            ID_BROWSE_MODELS,
        ],
    }
}

/// The shell's own controls, visible whichever page is showing.
pub const SHELL_CONTROLS: [i32; 5] = [
    ID_NAV_CHAT,
    ID_NAV_MODELS,
    ID_NAV_MONITOR,
    ID_NAV_SETTINGS,
    ID_STRIP_STOP,
];

/// One row of the settings page: the box, its label, and what empty means.
///
/// **Every setting says what it does and what leaving it blank will do.**
/// Hermes: *"nothing may be discovered by clicking"*. A box labelled `cache`
/// with no further word is a question, not a setting.
pub struct Field {
    pub id: i32,
    pub label: &'static str,
    pub hint: &'static str,
    pub group: &'static str,
}

pub const FIELDS: &[Field] = &[
    Field {
        id: ID_CACHE,
        label: "expert cache",
        hint: "GiB held for streamed experts. Empty: measured from free memory.",
        group: "Performance",
    },
    Field {
        id: ID_THREADS,
        label: "generation threads",
        hint: "Empty: measured. Generation wants 2-4, not all of them.",
        group: "Performance",
    },
    Field {
        id: ID_THREADS_BATCH,
        label: "prefill threads",
        hint: "Empty: measured. Prefill wants every core, unlike generation.",
        group: "Performance",
    },
    Field {
        id: ID_CONTEXT,
        label: "context",
        hint: "Tokens to keep. Empty: the model's own limit.",
        group: "Model defaults",
    },
    Field {
        id: ID_NGL,
        label: "GPU layers",
        hint: "Layers to offload. Empty: none. 99: all of them.",
        group: "Model defaults",
    },
    Field {
        id: ID_PORT,
        label: "port",
        hint: "Where the server listens, and what the endpoint shows.",
        group: "Server",
    },
    Field {
        id: ID_MODELS_DIR,
        label: "models folder",
        hint: "Empty: %USERPROFILE%\\.chaos\\models. Several, separated by ; are all searched.",
        group: "Paths",
    },
];

/// The two settings that are on or off rather than typed.
pub const TOGGLES: &[Field] = &[
    Field {
        id: ID_AUTO,
        label: "measure this machine",
        hint: "Pick device, offload and cache from the hardware, not from defaults.",
        group: "Model defaults",
    },
    Field {
        id: ID_FORCE,
        label: "allow unverified architectures",
        hint: "Runs a model never diffed against llama.cpp. It can produce fluent nonsense rather than an error.",
        group: "Model defaults",
    },
];

/// The settings groups, in the order they are drawn.
pub const GROUPS: [&str; 4] = ["Model defaults", "Performance", "Server", "Paths"];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// **The bug this whole module exists to prevent.** A control on two pages
    /// is a control that one page will hide while the other still needs it.
    #[test]
    fn every_control_has_exactly_one_home() {
        let mut seen: HashSet<i32> = HashSet::new();
        for p in PAGES {
            for &id in controls(p) {
                assert!(
                    seen.insert(id),
                    "control {id} appears on more than one page ({:?})",
                    p
                );
            }
        }
        for id in SHELL_CONTROLS {
            assert!(
                seen.insert(id),
                "shell control {id} is also claimed by a page"
            );
        }
    }

    /// A page with no controls and no painter is a blank screen. Monitor is
    /// deliberately painted, so it is the one page allowed to be empty here --
    /// spelled out so that an *accidentally* empty page fails.
    #[test]
    fn only_the_monitor_page_is_painted_rather_than_built() {
        for p in PAGES {
            if p == Page::Monitor {
                assert!(controls(p).is_empty(), "Monitor grew a control");
            } else {
                assert!(!controls(p).is_empty(), "{:?} has nothing on it", p);
            }
        }
    }

    /// Each page is reachable by rail button, by menu, and by accelerator.
    /// Hermes: *"a command may have keyboard, palette, and visible
    /// affordances, but they invoke the same action"*.
    #[test]
    fn every_page_is_reachable_three_ways() {
        let mut accels = HashSet::new();
        for p in PAGES {
            assert_eq!(page_of_nav(nav_id(p)), Some(p));
            let menu = match p {
                Page::Chat => IDM_PAGE_CHAT,
                Page::Models => IDM_PAGE_MODELS,
                Page::Monitor => IDM_PAGE_MONITOR,
                Page::Settings => IDM_PAGE_SETTINGS,
            };
            assert_eq!(page_of_menu(menu), Some(p));
            assert!(accels.insert(p.accel()), "{:?} shares an accelerator", p);
        }
    }

    /// Menu ids must not collide with control ids: `WM_COMMAND` delivers both
    /// through the same parameter, and an overlap would fire a button from a
    /// menu pick.
    #[test]
    fn menu_commands_cannot_be_mistaken_for_controls() {
        let controls: HashSet<i32> = PAGES
            .iter()
            .flat_map(|&p| super::controls(p).iter().copied())
            .chain(SHELL_CONTROLS)
            .collect();
        let menus = [
            IDM_API_KEY,
            IDM_TEST_CONNECTION,
            IDM_RESCAN,
            IDM_OPEN_MODELS_DIR,
            IDM_EXIT,
            IDM_LOAD,
            IDM_STOP,
            IDM_DOWNLOAD,
            IDM_DELETE,
            IDM_COPY_ENDPOINT,
            IDM_PAGE_CHAT,
            IDM_PAGE_MODELS,
            IDM_PAGE_MONITOR,
            IDM_PAGE_SETTINGS,
            IDM_THEME_LIGHT,
            IDM_THEME_DARK,
            IDM_MANUAL,
            IDM_RELEASES,
            IDM_CRASH_LOG,
            IDM_ABOUT,
        ];
        let mut seen = HashSet::new();
        for m in menus {
            assert!(!controls.contains(&m), "menu id {m} is also a control id");
            assert!(seen.insert(m), "menu id {m} is used twice");
        }
    }

    /// **Every setting in the file is on the page.** The old window exposed
    /// three of nine, which is how `threads_batch` and `context` became things
    /// you had to know to edit a text file to reach.
    #[test]
    fn the_settings_page_covers_the_whole_settings_file() {
        let on_page: HashSet<i32> = FIELDS.iter().chain(TOGGLES).map(|f| f.id).collect();
        for id in [
            ID_CACHE,
            ID_THREADS,
            ID_THREADS_BATCH,
            ID_PORT,
            ID_CONTEXT,
            ID_NGL,
            ID_MODELS_DIR,
            ID_AUTO,
            ID_FORCE,
        ] {
            assert!(on_page.contains(&id), "setting {id} has no row");
        }
        // Nine engine settings. The file holds a tenth, `mode`, which is a
        // view preference rather than something passed to `chaos-serve`; it
        // lives in the View menu, where a theme belongs.
        assert_eq!(on_page.len(), 9, "the page shows {} rows", on_page.len());
    }

    /// Every field belongs to a group that is actually drawn, or its row is
    /// built and never appears.
    #[test]
    fn every_field_sits_in_a_drawn_group() {
        for f in FIELDS.iter().chain(TOGGLES) {
            assert!(
                GROUPS.contains(&f.group),
                "{} is in group {:?}, which is never drawn",
                f.label,
                f.group
            );
            assert!(!f.hint.is_empty(), "{} has no hint", f.label);
        }
    }

    /// A label that shouts is a label that was written for a button. These are
    /// sentences in a page, so they read like sentences.
    #[test]
    fn field_hints_are_sentences() {
        for f in FIELDS.iter().chain(TOGGLES) {
            let h = f.hint;
            assert!(h.ends_with('.'), "{:?} does not end its hint", f.label);
            assert!(
                h.chars()
                    .next()
                    .is_some_and(|c| c.is_uppercase() || c == '%'),
                "{:?} hint does not start with a capital",
                f.label
            );
        }
    }

    /// The rail order is the accelerator order is the menu order. Three lists
    /// that disagree is three chances to be wrong.
    #[test]
    fn the_rail_order_is_the_accelerator_order() {
        for (i, p) in PAGES.iter().enumerate() {
            assert_eq!(p.index(), i);
            assert_eq!(p.accel(), b'1' + i as u8);
        }
    }

    /// Every page says what it is for, in one sentence, ending in a full stop.
    #[test]
    fn every_page_introduces_itself() {
        for p in PAGES {
            assert!(!p.title().is_empty());
            assert!(
                p.subtitle().ends_with('.'),
                "{:?} subtitle is not a sentence",
                p
            );
            assert_eq!(p.label(), p.label().to_uppercase());
        }
    }
}
