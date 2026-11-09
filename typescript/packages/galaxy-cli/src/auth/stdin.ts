import { AuthError } from "./types";
import { isPlainToken } from "./token-shape";
import { abortReason } from "./abort";

/** An API key is tens of characters; anything near this is not one. */
export const MAX_KEY_BYTES = 16 * 1024;

const TOO_LARGE = `That is more than ${MAX_KEY_BYTES / 1024} KiB, which is not an API key. Pass the key alone.`;

/** The person pressed Ctrl-C at the key prompt (raw mode turns it into a character, not SIGINT). */
export class InputCancelled extends Error {
  constructor() {
    super("Cancelled.");
    this.name = "InputCancelled";
  }
}

const NOT_PLAIN =
  "That is not an API key: it has spaces or characters outside printable ASCII. Paste the key alone.";

/** Refuse a key that cannot be one, without repeating it. Several lines are left to the caller's own message. */
function checkKeyShape(text: string): string {
  const key = text.trim();
  if (key && !/[\r\n]/.test(key) && !isPlainToken(key)) throw new AuthError("usage", NOT_PLAIN);
  return text;
}

/** Everything piped on stdin, up to `MAX_KEY_BYTES`; stops reading, rejecting, when `signal` aborts. */
export function readPiped(stdin: NodeJS.ReadableStream, signal: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    const cleanup = () => {
      stdin.removeListener("data", onData);
      stdin.removeListener("end", onEnd);
      stdin.removeListener("error", onError);
      signal.removeEventListener("abort", onAbort);
    };
    const settle = (fn: () => void) => {
      cleanup();
      stdin.pause();
      fn();
    };
    const onData = (chunk: Buffer | string) => {
      const buf = Buffer.from(chunk);
      size += buf.length;
      if (size > MAX_KEY_BYTES) {
        settle(() => {
          reject(new AuthError("usage", TOO_LARGE));
        });
        return;
      }
      chunks.push(buf);
    };
    const onEnd = () => {
      cleanup();
      try {
        resolve(checkKeyShape(Buffer.concat(chunks).toString("utf8")));
      } catch (e) {
        reject(e instanceof Error ? e : new Error("Could not read the key.", { cause: e }));
      }
    };
    const onError = (e: Error) => {
      settle(() => {
        reject(e);
      });
    };
    const onAbort = () => {
      settle(() => {
        reject(abortReason(signal));
      });
    };
    if (signal.aborted) {
      onAbort();
      return;
    }
    stdin.on("data", onData);
    stdin.on("end", onEnd);
    stdin.on("error", onError);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export type RawStdin = NodeJS.ReadableStream & { setRawMode(mode: boolean): unknown };

export function isRawCapable(stdin: NodeJS.ReadableStream): stdin is RawStdin {
  return typeof (stdin as Partial<RawStdin>).setRawMode === "function";
}

/**
 * Escape sequences a terminal sends for keys that are not text: CSI (`ESC [ ... final`, the arrow,
 * Home, End and Delete keys, and `ESC [200~` if a terminal brackets a paste on its own) and SS3
 * (`ESC O x`, which the same keys send in application mode). Bracketed paste is never switched on
 * here, so a paste arrives as plain text and needs nothing of its own.
 */
const KEY_SEQUENCES = /\x1b(?:\[[0-9;?]*[\x20-\x2f]*[\x40-\x7e]|O[\x40-\x7e])/g;

/**
 * One line typed at a terminal without echo. Raw mode is always switched off again, whatever
 * ends the read. Enter or Ctrl-D finishes (empty, if nothing was typed), Ctrl-C cancels,
 * backspace works.
 */
export function readSecret(stdin: RawStdin, signal: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    // Code points, so a backspace removes one character, and `bytes` is the UTF-8 size that
    // `MAX_KEY_BYTES` is a limit on.
    let typed: string[] = [];
    let bytes = 0;
    const ok = (t: string) => !t.trim() || isPlainToken(t.trim());
    const cleanup = () => {
      stdin.removeListener("data", onData);
      signal.removeEventListener("abort", onAbort);
      stdin.setRawMode(false);
      stdin.pause();
    };
    const settle = (fn: () => void) => {
      cleanup();
      fn();
    };
    const onAbort = () => {
      settle(() => {
        reject(abortReason(signal));
      });
    };
    const onData = (chunk: Buffer | string) => {
      const text = Buffer.from(chunk).toString("utf8").replace(KEY_SEQUENCES, "");
      for (const ch of text) {
        if (ch === "\x03") {
          settle(() => {
            reject(new InputCancelled());
          });
          return;
        }
        if (ch === "\r" || ch === "\n" || ch === "\x04") {
          settle(() => {
            const key = typed.join("");
            if (ok(key)) resolve(key);
            else reject(new AuthError("usage", NOT_PLAIN));
          });
          return;
        }
        if (ch === "\x7f" || ch === "\b") {
          const last = typed.pop();
          if (last !== undefined) bytes -= Buffer.byteLength(last);
        } else if (ch >= " ") {
          typed.push(ch);
          bytes += Buffer.byteLength(ch);
        }
        if (bytes > MAX_KEY_BYTES) {
          typed = [];
          settle(() => {
            reject(new AuthError("usage", TOO_LARGE));
          });
          return;
        }
      }
    };
    if (signal.aborted) {
      reject(abortReason(signal));
      return;
    }
    stdin.setRawMode(true);
    stdin.on("data", onData);
    stdin.resume();
    signal.addEventListener("abort", onAbort, { once: true });
  });
}
