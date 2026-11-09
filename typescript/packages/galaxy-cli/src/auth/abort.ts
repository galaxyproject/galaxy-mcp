/**
 * The reason a signal was aborted with, as an `Error`, for rejecting a promise with it.
 *
 * `signal.reason` is typed `any` because `abort(reason)` takes anything. Every abort here comes from
 * `abort()` with no argument, `AbortSignal.timeout` or `AbortSignal.any`, which all give a
 * `DOMException`, an `Error`, and that is passed on as it is. Anything else is wrapped, so a
 * rejection never carries a bare string or `undefined`.
 */
export function abortReason(signal: AbortSignal): Error {
  const reason: unknown = signal.reason;
  return reason instanceof Error ? reason : new Error("Aborted.", { cause: reason });
}

/** How long a revocation gets, discovery included. Logout and a login taking its token back share it. */
const REVOKE_TIMEOUT_MS = 5000;

/**
 * Run `fn` with a signal that aborts after `ms`, and reject with a timeout error when that happens
 * even if `fn` ignores the signal. A plain timer rather than `AbortSignal.timeout`, so tests can
 * fake it; cleared when `fn` settles, so it never keeps the process alive.
 *
 * It is deliberately not tied to any caller's signal: a revocation runs after a Ctrl-C too (a
 * login cancelled once its token exists must still take the token back), so the Ctrl-C's signal
 * would end it before it started.
 */
export async function withDeadline<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  ms: number = REVOKE_TIMEOUT_MS,
): Promise<T> {
  const control = new AbortController();
  let timer: NodeJS.Timeout | undefined;
  const timedOut = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      const e = new Error(`timed out after ${Math.round(ms / 1000)} s`);
      e.name = "TimeoutError";
      control.abort(e);
      reject(e);
    }, ms);
  });
  try {
    const work = fn(control.signal);
    // If the deadline wins, `work` is left to fail on its own aborted signal.
    work.catch(() => undefined);
    return await Promise.race([work, timedOut]);
  } finally {
    clearTimeout(timer);
  }
}
