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
