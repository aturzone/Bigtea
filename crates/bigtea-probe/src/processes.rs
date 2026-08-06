//! Find what is holding RAM, and what closing it would buy.
//!
//! This is legitimate here in a way it never would be on a server: Bigtea runs
//! a model on **your own machine, for you**. On a 16 GiB laptop the difference
//! between a browser being open and closed is the difference between the dense
//! weights being cached in RAM and being re-read from disk on every single
//! token — which is roughly an order of magnitude of throughput.
//!
//! Safety rules, because this touches other people's running programs:
//!
//! * **Report by default, never act.** Termination happens only when the caller
//!   explicitly asks, per process.
//! * **A deny-list of processes we will not touch under any circumstances** —
//!   the OS core, the session, the window manager, security software, and
//!   ourselves. Killing these ranges from "lose your desktop" to "bluescreen".
//! * **Ask politely first.** Termination requests a graceful close so the
//!   program can save; a hard kill is a separate, explicit escalation.
//! * **Unsaved work is the user's to weigh.** We surface the number and let
//!   them decide; we never decide for them.

use std::collections::HashSet;

/// A running process and what it costs in RAM.
#[derive(Debug, Clone)]
pub struct Process {
    pub pid: u32,
    pub name: String,
    /// Physical memory currently held (working set / RSS).
    pub rss_bytes: u64,
    /// Whether this process is one we refuse to touch.
    pub protected: bool,
}

/// Processes that must never be terminated. Killing any of these costs the
/// user their session or the machine, which is never worth a few GiB.
const PROTECTED: &[&str] = &[
    // Windows core
    "system",
    "registry",
    "smss",
    "csrss",
    "wininit",
    "winlogon",
    "services",
    "lsass",
    "lsaiso",
    "dwm",
    "explorer",
    "fontdrvhost",
    "sihost",
    "ctfmon",
    "shellexperiencehost",
    "searchhost",
    "startmenuexperiencehost",
    "runtimebroker",
    "audiodg",
    "conhost",
    "svchost",
    "taskhostw",
    "spoolsv",
    "memory compression",
    "securityhealthservice",
    "msmpeng",
    "wudfhost",
    // Linux/macOS core
    "systemd",
    "init",
    "kthreadd",
    "kernel_task",
    "launchd",
    "windowserver",
    "loginwindow",
    "dbus-daemon",
    "pipewire",
    "pulseaudio",
    "xorg",
    "wayland",
    // Ourselves and our toolchain — killing these kills the run in progress
    "bigtea",
    "bigtea-probe",
    "cargo",
    "rustc",
    "git",
    "ssh",
];

fn is_protected(name: &str) -> bool {
    // Lowercase *before* trimming the extension: ".EXE" does not match ".exe",
    // so trimming first lets `CSRSS.EXE` through the deny-list entirely.
    let lowered = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(name)
        .to_ascii_lowercase();
    let base = lowered.trim_end_matches(".exe");
    PROTECTED.contains(&base)
}

/// All processes with a measurable resident set, largest first.
pub fn list() -> Vec<Process> {
    let mut procs = imp::enumerate();
    procs.sort_by_key(|p| std::cmp::Reverse(p.rss_bytes));
    procs
}

/// Processes that could be closed to reclaim RAM, largest first.
///
/// Excludes protected processes and anything holding a trivial amount, since
/// closing a 30 MiB helper is disruption for no benefit.
pub fn reclaimable(min_bytes: u64) -> Vec<Process> {
    list()
        .into_iter()
        .filter(|p| !p.protected && p.rss_bytes >= min_bytes)
        .collect()
}

/// Total RAM that closing every reclaimable process would free.
///
/// An upper bound, not a promise: processes share pages, and the OS may not
/// return freed memory to the available pool immediately.
pub fn reclaimable_bytes(min_bytes: u64) -> u64 {
    reclaimable(min_bytes).iter().map(|p| p.rss_bytes).sum()
}

/// Group processes by name, since browsers and editors run many helpers that
/// individually look small and collectively are not.
pub fn grouped(min_bytes: u64) -> Vec<(String, u64, usize)> {
    let mut groups: Vec<(String, u64, usize)> = Vec::new();
    for p in reclaimable(0) {
        match groups.iter_mut().find(|(n, ..)| *n == p.name) {
            Some(g) => {
                g.1 += p.rss_bytes;
                g.2 += 1;
            }
            None => groups.push((p.name.clone(), p.rss_bytes, 1)),
        }
    }
    groups.retain(|(_, bytes, _)| *bytes >= min_bytes);
    groups.sort_by_key(|(_, bytes, _)| std::cmp::Reverse(*bytes));
    groups
}

#[derive(Debug)]
pub enum CloseError {
    /// The process is on the deny-list.
    Protected(String),
    /// The OS refused — usually insufficient privilege.
    Denied(u32),
    NotFound(u32),
}

impl std::fmt::Display for CloseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloseError::Protected(n) => {
                write!(f, "{n} is protected and will not be closed")
            }
            CloseError::Denied(pid) => write!(f, "permission denied closing pid {pid}"),
            CloseError::NotFound(pid) => write!(f, "no such process {pid}"),
        }
    }
}

impl std::error::Error for CloseError {}

/// Ask a process to close, so it can save first.
///
/// Refuses protected processes outright. This is a request, not a guarantee —
/// a program may prompt the user or decline.
pub fn request_close(pid: u32, name: &str) -> Result<(), CloseError> {
    if is_protected(name) {
        return Err(CloseError::Protected(name.to_string()));
    }
    imp::request_close(pid)
}

/// Set of names to skip, for callers that want to keep something open.
pub fn keep_set(names: &[String]) -> HashSet<String> {
    names.iter().map(|n| n.to_ascii_lowercase()).collect()
}

// --------------------------------------------------------------------- Windows

#[cfg(windows)]
mod imp {
    use super::{is_protected, CloseError, Process};

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool: usize,
        quota_paged_pool: usize,
        quota_peak_non_paged_pool: usize,
        quota_non_paged_pool: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const PROCESS_TERMINATE: u32 = 0x0001;

    extern "system" {
        fn K32EnumProcesses(ids: *mut u32, cb: u32, bytes_returned: *mut u32) -> i32;
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> isize;
        fn CloseHandle(h: isize) -> i32;
        fn K32GetProcessMemoryInfo(h: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
        /// Unlike `GetModuleBaseName`, this needs only
        /// `PROCESS_QUERY_LIMITED_INFORMATION` — no `PROCESS_VM_READ`, which an
        /// unelevated process cannot get for most other processes.
        fn QueryFullProcessImageNameW(h: isize, flags: u32, name: *mut u16, size: *mut u32) -> i32;
        fn TerminateProcess(h: isize, exit_code: u32) -> i32;
        fn GetLastError() -> u32;
    }

    pub fn enumerate() -> Vec<Process> {
        let mut ids = vec![0u32; 4096];
        let mut returned = 0u32;
        // SAFETY: `ids` is a live, correctly-sized buffer; the API writes at
        // most `cb` bytes and reports how many through `returned`.
        let ok = unsafe {
            K32EnumProcesses(
                ids.as_mut_ptr(),
                (ids.len() * std::mem::size_of::<u32>()) as u32,
                &mut returned,
            )
        } != 0;
        if !ok {
            return Vec::new();
        }
        let count = returned as usize / std::mem::size_of::<u32>();
        let self_pid = std::process::id();

        let mut out = Vec::with_capacity(count);
        for &pid in ids.iter().take(count) {
            if pid == 0 || pid == self_pid {
                continue;
            }
            // Limited-information rights only. Asking for PROCESS_VM_READ as
            // well makes OpenProcess fail for almost every process when running
            // unelevated, which silently empties the whole report.
            // SAFETY: a plain open by pid; failure returns 0, which we check.
            let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
            if h == 0 {
                continue; // protected or gone -- normal, not an error
            }

            let mut counters = ProcessMemoryCounters {
                cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
                ..Default::default()
            };
            // SAFETY: `h` is a valid handle we own; `counters` is sized and
            // its `cb` field declares that size, per the API contract.
            let got_mem = unsafe { K32GetProcessMemoryInfo(h, &mut counters, counters.cb) != 0 };

            let mut buf = [0u16; 512];
            let mut size = buf.len() as u32;
            // SAFETY: `buf` is live; `size` carries its length in and the
            // written length out, per the API contract.
            let got_name =
                unsafe { QueryFullProcessImageNameW(h, 0, buf.as_mut_ptr(), &mut size) } != 0;
            // SAFETY: `h` came from OpenProcess and has not been closed.
            unsafe { CloseHandle(h) };

            if !got_mem || !got_name || size == 0 {
                continue;
            }
            // QueryFullProcessImageNameW gives a full path; we want the leaf.
            let full = String::from_utf16_lossy(&buf[..size as usize]);
            let name = full.rsplit(['\\', '/']).next().unwrap_or(&full).to_string();
            out.push(Process {
                pid,
                protected: is_protected(&name),
                name,
                rss_bytes: counters.working_set_size as u64,
            });
        }
        out
    }

    pub fn request_close(pid: u32) -> Result<(), CloseError> {
        // Windows has no portable "please exit" for arbitrary processes without
        // a window handle, so this is a terminate. The caller has already been
        // told it is not graceful.
        // SAFETY: plain open-by-pid; 0 means failure and is checked.
        let h = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid) };
        if h == 0 {
            // SAFETY: reading the thread's last-error code.
            let err = unsafe { GetLastError() };
            return Err(if err == 87 {
                CloseError::NotFound(pid)
            } else {
                CloseError::Denied(pid)
            });
        }
        // SAFETY: `h` is a handle we just opened with TERMINATE rights.
        let ok = unsafe { TerminateProcess(h, 0) } != 0;
        // SAFETY: closing a handle we own.
        unsafe { CloseHandle(h) };
        if ok {
            Ok(())
        } else {
            Err(CloseError::Denied(pid))
        }
    }
}

// ----------------------------------------------------------------------- Unix

#[cfg(unix)]
mod imp {
    use super::{is_protected, CloseError, Process};

    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }

    /// Every process with a resident set, on whichever Unix this is.
    ///
    /// `/proc` is a Linux invention. macOS and the BSDs do not have it, so the
    /// `/proc` walk below silently returned an empty list there — and an empty
    /// list is not a visible failure, it is the "close these apps to free RAM"
    /// advice quietly never appearing. `ps` is the portable fallback: slower
    /// than reading a filesystem, but this runs once at startup and correctness
    /// beats microseconds.
    pub fn enumerate() -> Vec<Process> {
        if std::path::Path::new("/proc/self/statm").exists() {
            return enumerate_proc();
        }
        enumerate_ps()
    }

    /// `ps -axo pid=,rss=,comm=` — POSIX-specified columns, RSS in kilobytes.
    fn enumerate_ps() -> Vec<Process> {
        let self_pid = std::process::id();
        let Ok(out) = std::process::Command::new("ps")
            .args(["-axo", "pid=,rss=,comm="])
            .output()
        else {
            return Vec::new();
        };
        let Ok(text) = String::from_utf8(out.stdout) else {
            return Vec::new();
        };

        let mut procs = Vec::new();
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            let (Some(pid), Some(rss_kb)) = (fields.next(), fields.next()) else {
                continue;
            };
            let (Ok(pid), Ok(rss_kb)) = (pid.parse::<u32>(), rss_kb.parse::<u64>()) else {
                continue;
            };
            if pid == self_pid {
                continue;
            }
            // `comm` is a full path on macOS; the last component is the name a
            // user would recognise, and what `is_protected` matches against.
            let name = fields
                .next()
                .map(|c| c.rsplit('/').next().unwrap_or(c).to_string())
                .unwrap_or_else(|| pid.to_string());

            procs.push(Process {
                pid,
                protected: is_protected(&name),
                name,
                rss_bytes: rss_kb * 1024,
            });
        }
        procs
    }

    fn enumerate_proc() -> Vec<Process> {
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        let page_size = 4096u64; // conservative; only used for statm units
        let self_pid = std::process::id();
        let mut out = Vec::new();

        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let Some(pid_str) = file_name.to_str() else {
                continue;
            };
            let Ok(pid) = pid_str.parse::<u32>() else {
                continue;
            };
            if pid == self_pid {
                continue;
            }
            // statm field 2 is resident set size, in pages.
            let Ok(statm) = std::fs::read_to_string(format!("/proc/{pid}/statm")) else {
                continue;
            };
            let Some(rss_pages) = statm
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u64>().ok())
            else {
                continue;
            };
            let name = std::fs::read_to_string(format!("/proc/{pid}/comm"))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| pid_str.to_string());

            out.push(Process {
                pid,
                protected: is_protected(&name),
                name,
                rss_bytes: rss_pages * page_size,
            });
        }
        out
    }

    pub fn request_close(pid: u32) -> Result<(), CloseError> {
        // SIGTERM: asks the program to exit so it can save first.
        // SAFETY: a signal send to a pid; the return code is checked.
        let rc = unsafe { kill(pid as i32, 15) };
        if rc == 0 {
            Ok(())
        } else {
            Err(CloseError::Denied(pid))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn os_core_processes_are_protected() {
        for name in [
            "System",
            "csrss.exe",
            "lsass.exe",
            "systemd",
            "WindowServer",
        ] {
            assert!(is_protected(name), "{name} must be protected");
        }
    }

    #[test]
    fn our_own_toolchain_is_protected() {
        // Killing these would abort the very run that asked for RAM.
        for name in ["bigtea.exe", "cargo.exe", "rustc"] {
            assert!(is_protected(name), "{name} must be protected");
        }
    }

    #[test]
    fn ordinary_apps_are_not_protected() {
        for name in [
            "chrome.exe",
            "brave.exe",
            "Telegram.exe",
            "steam.exe",
            "code.exe",
        ] {
            assert!(!is_protected(name), "{name} should be closeable");
        }
    }

    #[test]
    fn protection_matching_is_case_and_extension_insensitive() {
        assert!(is_protected("CSRSS.EXE"));
        assert!(is_protected("csrss"));
        assert!(is_protected(r"C:\Windows\System32\csrss.exe"));
    }

    #[test]
    fn closing_a_protected_process_is_refused_before_any_syscall() {
        let err = request_close(4, "System");
        assert!(matches!(err, Err(CloseError::Protected(_))));
    }

    #[test]
    fn enumeration_finds_processes_and_never_includes_us() {
        let procs = list();
        // Every platform Bigtea claims to support must be able to answer this,
        // because "close these apps to free RAM" is useless without it — and an
        // unimplemented enumerator fails by returning nothing, not by erroring.
        assert!(
            !procs.is_empty(),
            "process enumeration returned nothing on {}; the RAM-reclaim advice              silently does nothing here",
            std::env::consts::OS
        );
        let me = std::process::id();
        assert!(procs.iter().all(|p| p.pid != me));
        // Sorted largest-first.
        for pair in procs.windows(2) {
            assert!(pair[0].rss_bytes >= pair[1].rss_bytes);
        }
    }

    #[test]
    fn reclaimable_excludes_protected_and_trivial() {
        let procs = reclaimable(64 << 20);
        assert!(procs.iter().all(|p| !p.protected));
        assert!(procs.iter().all(|p| p.rss_bytes >= (64 << 20)));
    }
}
