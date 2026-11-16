/** A stream gets one listener however many times `exitQuietlyOnEpipe` is called. */
const seen = new WeakSet<object>();

/**
 * Exit quietly when the reader of our output goes away.
 *
 * `gxy auth token | head -c1` closes the pipe after one byte; the next write then fails with
 * EPIPE, which Node raises as an `error` event, and with no listener that is an uncaught
 * exception: a stack trace on stderr, in a command whose arguments and output are secrets.
 *
 * Either stream, once its reader has gone, just goes quiet and the run carries on, and its exit
 * code still says how it went. For `auth token` that is 0, as the output was what the reader
 * wanted (`| head` is normal). For a command whose exit code is the report (`auth status`), a
 * closed stdout must not hide a failure behind a 0, and a login still running has not finished.
 */
export function exitQuietlyOnEpipe(
  streams: { stdout: NodeJS.WritableStream; stderr: NodeJS.WritableStream } = process,
): void {
  const watch = (stream: NodeJS.WritableStream, onEpipe: () => void) => {
    if (seen.has(stream)) return;
    seen.add(stream);
    stream.on("error", (e: NodeJS.ErrnoException) => {
      // A listener silences every stream error, so any other one is thrown again to stay as
      // loud as it was with none.
      if (e.code === "EPIPE") onEpipe();
      else throw e;
    });
  };
  const silence = (stream: NodeJS.WritableStream) => () => {
    // Later writes would only raise the same error again.
    stream.write = () => true;
  };
  watch(streams.stdout, silence(streams.stdout));
  watch(streams.stderr, silence(streams.stderr));
}
