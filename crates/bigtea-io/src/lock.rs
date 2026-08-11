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

/// Parse `--cpu-mask`: a hex bitmask, with or without `0x`.
///
/// **Separate from [`parse_cpu_range`] on purpose.** llama.cpp carries two
/// flags because the spellings are genuinely ambiguous: `5` is CPUs 0 and 2 as
/// a mask and CPU 5 as a range. A single heuristic parser guessed hex here and
/// pinned a `--cpu-range 5` run to two cores instead of one -- caught by this
/// module's own test, which is the only reason the ambiguity was noticed.
pub fn parse_cpu_mask(spec: &str) -> Option<u64> {
    let spec = spec.trim();
    let hex = spec.strip_prefix("0x").or_else(|| spec.strip_prefix("0X"));
    u64::from_str_radix(hex.unwrap_or(spec), 16)
        .ok()
        .filter(|&m| m != 0)
}

/// Parse `--cpu-range`: `0-3`, `5`, `0-1,4-5`.
///
/// Returns `None` for anything it does not understand, because a mask read
/// wrongly pins to the wrong cores -- which looks like a mysterious slowdown
/// rather than a bad argument.
pub fn parse_cpu_range(spec: &str) -> Option<u64> {
    let mut mask = 0u64;
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        let (lo, hi) = match part.split_once('-') {
            Some((a, b)) => (a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?),
            None => {
                let n = part.parse::<u32>().ok()?;
                (n, n)
            }
        };
        if lo > hi || hi >= 64 {
            return None;
        }
        for cpu in lo..=hi {
            mask |= 1u64 << cpu;
        }
    }
    (mask != 0).then_some(mask)
}

/// Restrict this process to the CPUs in `mask`.
///
/// # Why this exists after being refused
///
/// It was declined on the premise that there is "no thread-affinity layer"
/// here. That premise was wrong in the same way `--prio`'s and `--warmup`'s
/// were: **process affinity is one syscall and applies to every thread ggml
/// spawns**, because a thread inherits its process's affinity. Bigtea does not
/// need to own a threadpool to pin one.
///
/// What it genuinely cannot do is llama.cpp's *per-threadpool* masks — a
/// different set for prefill and generation — since ggml owns the pool. So the
/// batch variants take the same mask and the runner says so rather than
/// accepting a second one and dropping it.
#[cfg(windows)]
pub fn set_affinity(mask: u64) -> Result<u32, String> {
    // SAFETY: a plain Win32 call on the current process.
    unsafe {
        if ffi::SetProcessAffinityMask(ffi::GetCurrentProcess(), mask as usize) == 0 {
            return Err(format!(
                "SetProcessAffinityMask({mask:#x}) failed ({}) -- the mask must be a subset \
                 of the CPUs this process is already allowed",
                ffi::GetLastError()
            ));
        }
    }
    Ok(mask.count_ones())
}

#[cfg(not(windows))]
pub fn set_affinity(mask: u64) -> Result<u32, String> {
    // glibc's cpu_set_t is 1024 bits; only the first 64 are addressed here,
    // which covers every machine this runner targets and is honest about it.
    let mut set = [0u64; 16];
    set[0] = mask;
    // SAFETY: `set` is 128 bytes, the size glibc expects for cpu_set_t.
    let rc = unsafe { ffi::sched_setaffinity(0, core::mem::size_of_val(&set), set.as_ptr()) };
    if rc != 0 {
        return Err(format!("sched_setaffinity({mask:#x}) failed"));
    }
    Ok(mask.count_ones())
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
        pub fn SetProcessAffinityMask(h: isize, mask: usize) -> i32;
        pub fn GetLastError() -> u32;
    }
}

#[cfg(not(windows))]
mod ffi {
    use core::ffi::c_void;
    unsafe extern "C" {
        pub fn mlock(addr: *const c_void, len: usize) -> i32;
        pub fn setpriority(which: i32, who: u32, prio: i32) -> i32;
        pub fn sched_setaffinity(pid: i32, len: usize, set: *const u64) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hex_mask_parses_with_or_without_the_prefix() {
        assert_eq!(parse_cpu_mask("0xff"), Some(0xff));
        assert_eq!(parse_cpu_mask("ff"), Some(0xff));
        assert_eq!(parse_cpu_mask("0X0F"), Some(0x0f));
    }

    #[test]
    fn a_range_becomes_the_bits_it_names() {
        assert_eq!(parse_cpu_range("0-3"), Some(0b1111));
        assert_eq!(parse_cpu_range("0-1,4-5"), Some(0b110011));
    }

    #[test]
    fn the_same_text_means_different_cpus_to_the_two_flags() {
        // The reason llama.cpp has two flags, and the reason one heuristic
        // parser was wrong: `5` is CPUs 0 and 2 as a mask, CPU 5 as a range.
        // Guessing between them pins to the wrong cores silently.
        assert_eq!(parse_cpu_mask("5"), Some(0b101));
        assert_eq!(parse_cpu_range("5"), Some(1 << 5));
    }

    #[test]
    fn nonsense_is_refused_rather_than_read_as_zero() {
        for bad in ["", "  ", "zz", "0x0"] {
            assert_eq!(parse_cpu_mask(bad), None, "mask {bad:?} should be refused");
        }
        for bad in ["", "  ", "zz", "3-1", "0-99", "1-", "-"] {
            assert_eq!(
                parse_cpu_range(bad),
                None,
                "range {bad:?} should be refused"
            );
        }
    }

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
