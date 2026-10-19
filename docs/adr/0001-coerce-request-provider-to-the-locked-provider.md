# Coerce a request's Provider to the Locked Provider; do not reject it

When `DEEPWIKI_LOCK_PROVIDER` pins the deployment to one Provider, a generation
request's `provider` field is deliberately **ignored** and replaced with the
locked Provider — for an omitted Provider as much as for an explicitly different
one. We chose coercion over rejection because the field is a UI artefact, not
caller intent: the request schema defaults `provider` to `google` and the
frontend drops empty selectors, so a caller that never chose a Provider still
arrives carrying `google`. Rejecting on mismatch would therefore break the very
requests the lock exists to fix, and would force every client to learn the lock.
Coercion lets one environment variable pin the deployment with no client change
and no silent fallback to Google.

## Consequences

- A request that names a different Provider succeeds but is served by the locked
  one. This is intended, and is **not** a bug to "fix" — the alternative
  (honouring it, or erroring) is what the lock exists to prevent.
- The lock is enforced only on the backend. The frontend hiding the selector
  (a separate change) is belt-and-braces; the backend resolver is the
  load-bearing part.
- The `provider` field stays in the request contract (unchanged shape), so the
  lock is reversible by unsetting the variable — no client protocol change.
