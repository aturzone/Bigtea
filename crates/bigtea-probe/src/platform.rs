//! Per-OS facts: RAM and filesystem capacity.
//!
//! These are the handful of calls the standard library does not expose. Each
//! `unsafe` block is a single FFI call with the buffer it writes into owned
//! and sized locally, so the unsafety is bounded to "the OS honours its own
//! documented contract".

use std::path::Path;

// --------------------------------------------------------------------- Windows

#[cfg(windows)]
mod imp {
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        fn GetDiskFreeSpaceExW(
            directory_name: *const u16,
            free_bytes_available_to_caller: *mut u64,
            total_number_of_bytes: *mut u64,
            total_number_of_free_bytes: *mut u64,
        ) -> i32;
    }

    pub fn ram() -> (Option<u64>, Option<u64>, String) {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `status` is a correctly-sized, correctly-initialised struct
        // whose `length` field tells the API its size, per its contract.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
        if !ok {
            return (None, None, "GlobalMemoryStatusEx failed".into());
        }
        (
            Some(status.total_phys),
            Some(status.avail_phys),
            "GlobalMemoryStatusEx".into(),
        )
    }

    pub fn disk_space(path: &Path) -> Option<(u64, u64)> {
        // The API wants a directory; a file path would fail.
        let dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent()?.to_path_buf()
        };
        let mut wide: Vec<u16> = dir.as_os_str().encode_wide().collect();
        wide.push(0);

        let (mut free_to_caller, mut total, mut total_free) = (0u64, 0u64, 0u64);
        // SAFETY: `wide` is NUL-terminated and outlives the call; the three
        // out-pointers reference locals of the right type.
        let ok = unsafe {
            GetDiskFreeSpaceExW(wide.as_ptr(), &mut free_to_caller, &mut total, &mut total_free)
        } != 0;
        if !ok {
            return None;
        }
        // Report the quota-aware figure: what this user may actually write.
        Some((total, free_to_caller))
    }

    pub fn os_description() -> String {
        "Windows".into()
    }
}

// ----------------------------------------------------------------------- Unix

#[cfg(unix)]
mod imp {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    pub fn ram() -> (Option<u64>, Option<u64>, String) {
        // Linux: /proc/meminfo is authoritative and cheap.
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            let mut total = None;
            let mut available = None;
            for line in text.lines() {
                let Some((key, rest)) = line.split_once(':') else { continue };
                let kb: Option<u64> = rest.split_whitespace().next().and_then(|v| v.parse().ok());
                match key {
                    "MemTotal" => total = kb.map(|v| v * 1024),
                    // MemAvailable accounts for reclaimable cache, which is what
                    // actually matters for "can I hold weights here".
                    "MemAvailable" => available = kb.map(|v| v * 1024),
                    _ => {}
                }
            }
            if total.is_some() {
                return (total, available, "/proc/meminfo".into());
            }
        }

        // macOS and the BSDs: sysctl gives total. "Available" has no honest
        // single-number answer under a memory compressor, so report None
        // rather than invent one.
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            if let Ok(text) = String::from_utf8(out.stdout) {
                if let Ok(bytes) = text.trim().parse::<u64>() {
                    return (Some(bytes), None, "sysctl hw.memsize".into());
                }
            }
        }
        (None, None, "no supported source".into())
    }

    #[repr(C)]
    #[derive(Default)]
    struct StatVfs {
        f_bsize: u64,
        f_frsize: u64,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        f_files: u64,
        f_ffree: u64,
        f_favail: u64,
        f_fsid: u64,
        f_flag: u64,
        f_namemax: u64,
        f_spare: [u64; 6],
    }

    extern "C" {
        fn statvfs(path: *const std::ffi::c_char, buf: *mut StatVfs) -> i32;
    }

    pub fn disk_space(path: &Path) -> Option<(u64, u64)> {
        let dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent()?.to_path_buf()
        };
        let c = CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut st = StatVfs::default();
        // SAFETY: `c` is a NUL-terminated path alive across the call, and `st`
        // is a locally-owned struct of the layout the C API expects.
        if unsafe { statvfs(c.as_ptr(), &mut st) } != 0 {
            return None;
        }
        // f_frsize is the fragment size the block counts are expressed in.
        let unit = if st.f_frsize > 0 { st.f_frsize } else { st.f_bsize };
        // f_bavail, not f_bfree: blocks available to an unprivileged user.
        Some((st.f_blocks * unit, st.f_bavail * unit))
    }

    pub fn os_description() -> String {
        std::env::consts::OS.to_string()
    }
}

pub fn ram() -> (Option<u64>, Option<u64>, String) {
    imp::ram()
}

pub fn disk_space(path: &Path) -> Option<(u64, u64)> {
    imp::disk_space(path)
}

pub fn os_description() -> String {
    imp::os_description()
}
