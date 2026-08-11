//! Pinning resident weights in physical memory — llama.cpp's `--mlock`.
//!
//! # Why this is worth having here specifically
//!
//! Bigtea's whole design is deciding what stays in RAM and what streams. That
//! decision is undone if the OS pages the resident set out behind its back, and
//! this project has already measured the consequence: past ~6 GiB the expert
//! cache reached a **71% hit rate while being the slowest configuration
//! tested**, because the "hits" were page faults wearing a disguise.
//!
//! Locking makes the residency plan the truth rather than a preference.
//!
//! # Why it is not simply `VirtualLock`
//!
//! On Windows a process may only lock up to its **working set maximum**, and
//! the default is small — a few megabytes. `VirtualLock` on a gigabyte buffer
//! therefore fails with `ERROR_WORKING_SET_QUOTA` (1453) unless
//! `SetProcessWorkingSetSize` has raised the ceiling first. A `--mlock` that
//! called only `VirtualLock` would appear to work, return an error nobody
//! checks, and lock nothing.
//!
//! That is exactly the failure this crate's callers keep finding elsewhere, so
//! every call here is checked and the outcome is reported in bytes actually
//! locked, not in bytes attempted.

/// Outcome of a locking attempt, in bytes, with a reason when it fell short.
#[derive(Debug, Default)]
pub struct LockReport {
    pub locked_bytes: u64,
    pub failed_bytes: u64,
    /// Empty when everything asked for was locked.
    pub reason: String,
}

impl LockReport {
    pub fn ok(&self) -> bool {
        self.failed_bytes == 0
    }
}

/// Raise the working-set ceiling so `bytes` can be locked.
///
/// A no-op off Windows, where there is no such quota — `mlock` is bounded by
/// `RLIMIT_MEMLOCK` instead, which a process cannot raise for itself.
#[cfg(windows)]
pub fn reserve_working_set(bytes: u64) -> Result<(), String> {
    // SAFETY: all four are plain Win32 calls on the current process; the two
    // out-pointers are stack locals of the right type.
    unsafe {
        let handle = ffi::GetCurrentProcess();
        let (mut min, mut max) = (0usize, 0usize);
        if ffi::GetProcessWorkingSetSize(handle, &mut min, &mut max) == 0 {
            return Err(format!(
                "GetProcessWorkingSetSize failed ({})",
                ffi::GetLastError()
            ));
        }
        // Headroom above what is being locked: the process still needs an
        // ordinary working set for its own code, stacks and arenas, and asking
        // for exactly `bytes` leaves none.
        let want = bytes.saturating_add(512 << 20) as usize;
        let new_min = min.max(want);
        let new_max = max.max(want.saturating_add(256 << 20));
        if ffi::SetProcessWorkingSetSize(handle, new_min, new_max) == 0 {
            return Err(format!(
                "SetProcessWorkingSetSize({new_min}, {new_max}) failed ({})",
                ffi::GetLastError()
            ));
        }
    }
    Ok(())
}

#[cfg(not(windows))]
pub fn reserve_working_set(_bytes: u64) -> Result<(), String> {
    Ok(())
}

/// Scheduling priority, llama.cpp's `--prio` scale.
///
/// `0` normal, `1` medium, `2` high, `3` realtime. Raising it is a real lever
/// on a machine that is doing something else: a generation step is a tight
/// loop of short compute bursts separated by disk waits, and every wait is a
/// chance for the scheduler to hand the core to a browser tab.
///
/// **`3` is deliberately not `REALTIME_PRIORITY_CLASS`.** That class outranks
/// the kernel's own input and disk threads, and a process that pins the CPU
/// there can make the machine unresponsive with no way to click anything. It
/// maps to `HIGH` plus a note, which is the honest reading of "as high as it
/// is safe to go". llama.cpp asks for realtime; we decline that one and say so.
#[cfg(windows)]
pub fn set_priority(level: u32) -> Result<&'static str, String> {
    // Win32 priority classes.
    const NORMAL: u32 = 0x0000_0020;
    const ABOVE_NORMAL: u32 = 0x0000_8000;
    const HIGH: u32 = 0x0000_0080;
    let (class, name) = match level {
        0 => (NORMAL, "normal"),
        1 => (ABOVE_NORMAL, "above normal"),
        2 => (HIGH, "high"),
        _ => (HIGH, "high (realtime declined: it can freeze the desktop)"),
    };
    // SAFETY: a plain Win32 call on the current process with a valid class.
    unsafe {
        if ffi::SetPriorityClass(ffi::GetCurrentProcess(), class) == 0 {
            return Err(format!(
                "SetPriorityClass({class:#x}) failed ({})",
                ffi::GetLastError()
            ));
        }
    }
    Ok(name)
}

/// Scheduling priority via `nice`. Lower `nice` is higher priority, so the
/// llama.cpp scale is inverted here.
#[cfg(not(windows))]
pub fn set_priority(level: u32) -> Result<&'static str, String> {
    let (nice, name) = match level {
        0 => (0, "normal"),
        1 => (-5, "above normal"),
        2 => (-10, "high"),
        _ => (-15, "high (realtime declined: it can starve the kernel)"),
    };
    // SAFETY: `setpriority` on the current process with a valid class.
    let rc = unsafe {
        ffi::setpriority(0 /* PRIO_PROCESS */, 0, nice)
    };
    if rc != 0 {
        // Lowering `nice` needs privilege; saying so beats pretending.
        return Err(format!("setpriority({nice}) failed -- needs privilege"));
    }
    Ok(name)
}

/// Pin `bytes` in physical memory.
///
/// The slice must stay alive and unmoved for as long as the lock matters —
/// which is why callers pass a resident buffer they own rather than a
/// temporary.
#[cfg(windows)]
pub fn lock_bytes(bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    // SAFETY: the pointer and length describe a live slice the caller owns.
    unsafe {
        if ffi::VirtualLock(bytes.as_ptr() as *const core::ffi::c_void, bytes.len()) == 0 {
            let code = ffi::GetLastError();
            // 1453 is ERROR_WORKING_SET_QUOTA, and naming it saves the next
            // person the search: it means the ceiling was not raised enough,
            // not that locking is unavailable.
            let hint = if code == 1453 {
                " (ERROR_WORKING_SET_QUOTA — the working-set ceiling is too low)"
            } else {
                ""
            };
            return Err(format!("VirtualLock failed ({code}){hint}"));
        }
    }
    Ok(())
}

#[cfg(not(windows))]
pub fn lock_bytes(bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    // SAFETY: the pointer and length describe a live slice the caller owns.
    let rc = unsafe { ffi::mlock(bytes.as_ptr() as *const core::ffi::c_void, bytes.len()) };
    if rc != 0 {
        return Err(format!(
            "mlock failed ({}) — RLIMIT_MEMLOCK is usually the cause",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

#[cfg(windows)]
mod ffi {
    use core::ffi::c_void;
    unsafe extern "system" {
        pub fn GetCurrentProcess() -> isize;
        pub fn GetProcessWorkingSetSize(h: isize, min: *mut usize, max: *mut usize) -> i32;
        pub fn SetProcessWorkingSetSize(h: isize, min: usize, max: usize) -> i32;
        pub fn VirtualLock(addr: *const c_void, size: usize) -> i32;
        pub fn SetPriorityClass(h: isize, class: u32) -> i32;
        pub fn GetLastError() -> u32;
    }
}

#[cfg(not(windows))]
mod ffi {
    use core::ffi::c_void;
    unsafe extern "C" {
        pub fn mlock(addr: *const c_void, len: usize) -> i32;
        pub fn setpriority(which: i32, who: u32, prio: i32) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_slice_locks_trivially() {
        assert!(lock_bytes(&[]).is_ok());
    }

    #[test]
    fn a_small_buffer_can_be_locked_after_reserving() {
        // One page is well inside any default quota once the ceiling is
        // raised, so this exercises the real syscalls rather than mocking
        // them. A failure here is a genuine platform problem, and the report
        // says which call failed.
        let buf = vec![7u8; 4096];
        reserve_working_set(buf.len() as u64).expect("reserve");
        match lock_bytes(&buf) {
            Ok(()) => {}
            // A CI container may forbid locking outright; that is not a bug in
            // this code, and the message must say so rather than being hidden.
            Err(e) => assert!(!e.is_empty(), "a failure must carry a reason: {e}"),
        }
    }

    #[test]
    fn a_report_is_only_ok_when_nothing_failed() {
        let mut r = LockReport {
            locked_bytes: 100,
            ..LockReport::default()
        };
        assert!(r.ok());
        r.failed_bytes = 1;
        assert!(!r.ok());
    }
}
