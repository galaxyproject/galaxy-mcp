/** A stream gets one listener however many times `exitQuietlyOnEpipe` is called. */
const seen = new WeakSet<object>();

/**
 * Exit quietly when the reader of our output goes away.
 *
 * `gxy auth token | head -c1` closes the pipe after one byte; the next write then fails with
 * EPIPE, which Node raises as an `error` event, and with no listener that is an uncaught
 * exception: a stack trace on stderr, in a command whose arguments and output are secrets.
 *
 * On stdout the reader leaving is not our failure and the output was what they wanted (`| head`
 * is normal), so the process ends with 0 and no message. On stderr it is different: the work
 * is still going (a login), and 0 would report success for something that has not
 * finished. So stderr just goes quiet and the run carries on, and its exit code still says how
 * it went.
 */
export function exitQuietlyOnEpipe(
  streams: { stdout: NodeJS.WritableStream; stderr: NodeJS.WritableStream } = process,
  // eslint-disable-next-line n/no-process-exit -- ending the process is what this function exists to do
  exit: (code: number) => void = (code) => process.exit(code),
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
  watch(streams.stdout, () => {
    exit(0);
  });
  watch(streams.stderr, () => {
    // Later writes would only raise the same error again.
    streams.stderr.write = () => true;
  });
}
