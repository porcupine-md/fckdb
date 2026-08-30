//! Turning an engine failure into a status code without reading its prose.
//!
//! The server used to decide by keyword: `msg.contains("declared")`, sixteen
//! phrases and growing. That defaults a new validation message to 500 until
//! somebody notices, and rewording any `bail!` silently changes the status.
//!
//! The discriminator is already there and does not need to be written down: a
//! genuine fault arrives carrying the type that produced it — `object_store::Error`
//! from a GET that failed, `io::Error` from a socket — somewhere in its chain. A
//! caller mistake is an `anyhow!` this crate raised with nothing underneath,
//! because that is what validation code does. So the default is 400, faults are
//! recognised by type, and only the handful of failures that are neither get an
//! explicit [`Kind`].

use std::fmt;

/// A failure whose status is not the default. Attach with [`kinded`] or the
/// `bail_kind!` macro; read back with [`kind_of`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Recognised, deliberately not built. 501.
    Unimplemented,
    /// True right now, may not be in a moment. Retry is meaningful. 503.
    Unavailable,
    /// The caller asked to create something that is already there. 409.
    Conflict,
    /// Somebody else's service failed. Neither side's fault. 502.
    Upstream,
    /// Ours. For the faults that carry no telltale type of their own — a
    /// background task that died, an invariant that did not hold — and so would
    /// otherwise fall through to the caller-mistake default.
    Internal,
}

/// An error carrying a [`Kind`]. Displays as its message alone, so tagging a
/// failure does not change what the caller reads.
#[derive(Debug)]
pub struct Kinded {
    pub kind: Kind,
    pub msg: String,
}

impl fmt::Display for Kinded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for Kinded {}

pub fn kinded(kind: Kind, msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Kinded { kind, msg: msg.into() })
}

/// A failure flattened for transport, keeping the bit that decides its status.
///
/// `anyhow::Error` is neither `Clone` nor cheap to share, and the group committer
/// has to hand one failure to every waiter in the batch — so it stringifies. That
/// throws away the source type, which is precisely what tells a storage fault
/// from a caller's mistake. Carrying the verdict alongside the message keeps the
/// distinction across the channel.
#[derive(Debug, Clone)]
pub struct Flat {
    pub kind: Option<Kind>,
    pub fault: bool,
    pub msg: String,
}

impl Flat {
    pub fn new(e: &anyhow::Error) -> Self {
        Self { kind: kind_of(e), fault: is_fault(e), msg: format!("{e:#}") }
    }

    pub fn into_error(self) -> anyhow::Error {
        match self.kind {
            Some(k) => kinded(k, self.msg),
            // The type is gone, so say plainly what it told us.
            None if self.fault => kinded(Kind::Internal, self.msg),
            None => anyhow::anyhow!(self.msg),
        }
    }
}

/// The kind attached anywhere in the chain, if any.
pub fn kind_of(e: &anyhow::Error) -> Option<Kind> {
    e.chain().find_map(|c| c.downcast_ref::<Kinded>()).map(|k| k.kind)
}

/// Whether this failure is the process's fault rather than the caller's.
///
/// True when something that actually talks to the world is in the chain. A
/// `bail!` raised by this crate has no such source and is therefore a caller
/// mistake — which is the default, so validation code never has to say so.
pub fn is_fault(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        c.is::<object_store::Error>()
            || c.is::<std::io::Error>()
            || c.is::<serde_json::Error>()

    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn a_plain_bail_is_the_callers_mistake() {
        let e = anyhow!("document 9 has 999 dimensions, this namespace has 3");
        assert!(!is_fault(&e));
        assert_eq!(kind_of(&e), None);
    }

    #[test]
    fn an_io_failure_is_a_fault_even_under_context() {
        let io = std::io::Error::other("connection reset");
        let e = anyhow::Error::new(io).context("reading segment").context("query");
        assert!(is_fault(&e), "a fault wrapped in context is still a fault");
    }

    #[test]
    fn object_store_failures_are_faults() {
        let e = anyhow::Error::new(object_store::Error::NotFound {
            path: "ns/x/manifest".into(),
            source: Box::new(std::io::Error::other("gone")),
        });
        assert!(is_fault(&e));
    }

    #[test]
    fn a_kind_survives_context_and_does_not_change_the_message() {
        let e = kinded(Kind::Unavailable, "unindexed WAL is over the scan cap");
        assert_eq!(format!("{e}"), "unindexed WAL is over the scan cap");
        assert_eq!(kind_of(&e), Some(Kind::Unavailable));
        let wrapped = e.context("querying");
        assert_eq!(kind_of(&wrapped), Some(Kind::Unavailable), "context hid the kind");
    }

    #[test]
    fn an_internal_kind_overrides_the_caller_mistake_default() {
        // A dead committer surfaces as a oneshot RecvError, which is not a type
        // that says "I talked to the world" — so without the tag it would read
        // as the caller's mistake.
        let e = kinded(Kind::Internal, "committer dropped the request");
        assert!(!is_fault(&e), "RecvError is not a fault type");
        assert_eq!(kind_of(&e), Some(Kind::Internal));
    }

    #[test]
    fn flattening_preserves_the_verdict_across_a_channel() {
        // The committer stringifies, so without this a storage failure during a
        // write would arrive indistinguishable from a caller's bad JSON.
        let io = anyhow::Error::new(std::io::Error::other("connection reset"))
            .context("PUT ns/x/wal/0001");
        let back = Flat::new(&io).into_error();
        assert!(
            is_fault(&back) || kind_of(&back) == Some(Kind::Internal),
            "a fault flattened to a string stopped looking like a fault"
        );
        assert!(back.to_string().contains("PUT ns/x/wal/0001"));

        let mine = anyhow::anyhow!("document 9 has 999 dimensions");
        let back = Flat::new(&mine).into_error();
        assert!(!is_fault(&back) && kind_of(&back).is_none(), "a caller mistake became a fault");

        let tagged = kinded(Kind::Unavailable, "CAS contention");
        assert_eq!(kind_of(&Flat::new(&tagged).into_error()), Some(Kind::Unavailable));
    }

    #[test]
    fn a_kind_beats_the_fault_default() {
        // An upstream HTTP failure carries io::Error but is not our fault.
        let e = kinded(Kind::Upstream, "embedding endpoint returned 500");
        assert_eq!(kind_of(&e), Some(Kind::Upstream));
    }
}
