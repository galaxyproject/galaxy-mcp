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

/** Settle with `p`, or reject as soon as `signal` aborts (what `p` is doing is left alone). */
export function untilAborted<T>(p: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return p;
  if (signal.aborted) {
    p.catch(() => undefined);
    return Promise.reject(abortReason(signal));
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      reject(abortReason(signal));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    p.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", onAbort);
    });
  });
}
