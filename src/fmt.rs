//! Small display-formatting helpers shared by CLI subcommands that print a
//! table of images/models (`list`, `ps`) — kept in one place rather than
//! duplicated per-file so the two commands' NAME/ID/SIZE columns always
//! render identically for the same underlying digest/byte count.

use std::time::SystemTime;

/// First 12 hex chars of a `sha256:...` digest, matching `docker images`'s
/// convention (and Ollama's `ollama ps`/`ollama list`, which truncate to 12
/// as well).
pub fn short_id(digest: &str) -> String {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    hex.chars().take(12).collect()
}

/// Decimal (not binary) byte units, matching `llmman list`'s existing
/// convention — GB/MB/kB, not GiB/MiB/KiB.
pub fn human_size(bytes: u64) -> String {
    const GB: u64 = 1_000_000_000;
    const MB: u64 = 1_000_000;
    const KB: u64 = 1_000;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} kB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Relative-time string ("5 minutes ago", "yesterday", ...) for a duration
/// expressed in seconds-in-the-past. Shared core of [`relative_time`] and
/// [`relative_time_rfc3339`], which just compute `secs` differently.
fn relative_time_secs(secs: u64) -> String {
    match secs {
        s if s < 60 => "just now".into(),
        s if s < 3600 => format!("{} minutes ago", s / 60),
        s if s < 86400 => format!("{} hours ago", s / 3600),
        s if s < 86400 * 2 => "yesterday".into(),
        s if s < 86400 * 7 => format!("{} days ago", s / 86400),
        s if s < 86400 * 14 => "1 week ago".into(),
        s if s < 86400 * 30 => format!("{} weeks ago", s / (86400 * 7)),
        s if s < 86400 * 60 => "1 month ago".into(),
        s if s < 86400 * 365 => format!("{} months ago", s / (86400 * 30)),
        s if s < 86400 * 730 => "1 year ago".into(),
        s => format!("{} years ago", s / (86400 * 365)),
    }
}

/// Relative-time string for a `SystemTime` (e.g. filesystem mtime) — used
/// by `llmman list`'s MODIFIED column.
pub fn relative_time(t: Option<SystemTime>) -> String {
    let secs = match t {
        Some(t) => SystemTime::now().duration_since(t).unwrap_or_default().as_secs(),
        None => return "unknown".into(),
    };
    relative_time_secs(secs)
}

/// Relative-time string for an RFC 3339 timestamp (e.g. `llmman serve`'s
/// own `now_rfc3339()` output) — used by `llmman ps`'s STARTED column.
pub fn relative_time_rfc3339(s: &str) -> String {
    let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(s) else {
        return "unknown".into();
    };
    let secs = (chrono::Utc::now() - parsed.with_timezone(&chrono::Utc))
        .num_seconds()
        .max(0) as u64;
    relative_time_secs(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_id_truncates_to_12_hex_chars() {
        assert_eq!(
            short_id("sha256:0123456789abcdef0123456789abcdef"),
            "0123456789ab"
        );
        // No prefix: still truncates.
        assert_eq!(short_id("0123456789abcdef"), "0123456789ab");
    }

    #[test]
    fn human_size_picks_the_right_unit() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1_500), "1.5 kB");
        assert_eq!(human_size(1_500_000), "1.5 MB");
        assert_eq!(human_size(1_500_000_000), "1.5 GB");
    }

    #[test]
    fn relative_time_rfc3339_handles_recent_and_invalid_timestamps() {
        let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        assert_eq!(relative_time_rfc3339(&now), "just now");
        assert_eq!(relative_time_rfc3339("not a timestamp"), "unknown");
    }
}
