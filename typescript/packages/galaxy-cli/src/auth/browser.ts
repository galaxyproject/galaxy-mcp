import { spawn } from "node:child_process";

type Env = Record<string, string | undefined>;

/** How long a launcher gets to start before we stop waiting; login never blocks on a browser. */
export const SPAWN_TIMEOUT_MS = 3000;

/**
 * How long a launcher that has started may take to fail before we call it a success. One that
 * exits non-zero inside this (`BROWSER=false`, a text browser with no terminal, `xdg-open` with
 * nothing to open the URL) did not open anything; one still running is a browser that stays open.
 */
export const EARLY_EXIT_MS = 500;

/**
 * Split a `$BROWSER` command into words. No shell is involved, so only the quoting a person needs
 * for a path with spaces is understood: `'...'` and `"..."` group words and are dropped, with no
 * escapes inside them. `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" %s` is
 * therefore one command and one argument. Anything fancier (`$VAR`, `~`, `\ `) goes in a wrapper
 * script, which `$BROWSER` can point at.
 */
export function splitCommand(command: string): string[] {
  const words: string[] = [];
  let word = "";
  let started = false; // distinguishes an empty quoted word ("") from no word
  let quote: string | undefined;
  for (const ch of command) {
    if (quote) {
      if (ch === quote) quote = undefined;
      else word += ch;
    } else if (ch === '"' || ch === "'") {
      quote = ch;
      started = true;
    } else if (ch.trim() === "") {
      if (started) words.push(word);
      word = "";
      started = false;
    } else {
      word += ch;
      started = true;
    }
  }
  if (started) words.push(word);
  return words;
}

/** `$BROWSER` as the convention has it: a colon-separated list of commands to try in order. */
function browserCandidates(env: Env): string[] {
  return (env.BROWSER ?? "")
    .split(":")
    .map((c) => c.trim())
    .filter(Boolean);
}

/**
 * Whether there is any chance of opening a browser: `$BROWSER` set, else always off Linux,
 * and on Linux a display. `$BROWSER` comes first because VS Code's remote and devcontainer
 * sessions set it to a helper that forwards the URL to the browser on the user's own
 * machine, where there is often no `DISPLAY` at all.
 */
export function canOpenBrowser(
  env: Env = process.env,
  platform: NodeJS.Platform = process.platform,
): boolean {
  if (browserCandidates(env).length > 0) return true;
  if (platform === "linux") return Boolean(env.DISPLAY || env.WAYLAND_DISPLAY);
  return true;
}

/**
 * Open a URL in the user's browser, detached. The URL is an argv entry, never part of a
 * shell string: it came from the server, and a `;` or a backtick in it must not become a
 * command. On Windows, `rundll32 url.dll,FileProtocolHandler` takes it the same way;
 * `cmd /c start` would hand `&` in a URL to cmd's parser.
 *
 * Order: each `$BROWSER` candidate (a `%s` in one is replaced by the URL, otherwise the URL
 * is appended; no shell is involved, and quotes group words, see `splitCommand`), then the
 * platform opener. The first launcher that starts and does not fail at once wins. Resolves false
 * when none did, and never waits longer than `SPAWN_TIMEOUT_MS` to start or `EARLY_EXIT_MS` to
 * fail for any one of them. A launcher's stdio is ignored, so a browser that needs a terminal
 * (lynx, w3m) cannot work here: it exits non-zero and the next one is tried.
 */
export async function openBrowser(
  url: string,
  env: Env = process.env,
  platform: NodeJS.Platform = process.platform,
  spawnImpl: typeof spawn = spawn,
): Promise<boolean> {
  // Only web pages: a launcher given `file:` or an app's own scheme would run whatever handles it.
  if (!/^https?:\/\//i.test(url)) return false;
  const launchers: [string, string[]][] = browserCandidates(env).map((candidate) => {
    const [command = candidate, ...args] = splitCommand(candidate);
    return candidate.includes("%s")
      ? [command, args.map((p) => p.replaceAll("%s", url))]
      : [command, [...args, url]];
  });
  if (platform !== "linux" || env.DISPLAY || env.WAYLAND_DISPLAY) {
    launchers.push(
      platform === "darwin"
        ? ["open", [url]]
        : platform === "win32"
          ? ["rundll32", ["url.dll,FileProtocolHandler", url]]
          : ["xdg-open", [url]],
    );
  }
  for (const [cmd, args] of launchers) {
    if (await launch(spawnImpl, cmd, args)) return true;
  }
  return false;
}

function launch(spawnImpl: typeof spawn, cmd: string, args: string[]): Promise<boolean> {
  return new Promise((resolve) => {
    let timer: NodeJS.Timeout | undefined;
    let started = false;
    const done = (ok: boolean) => {
      clearTimeout(timer);
      resolve(ok);
    };
    try {
      // Detached with no stdio, and unref'd once it starts, so a browser that stays open does not
      // keep `gxy` running after login is done.
      const child = spawnImpl(cmd, args, { detached: true, stdio: "ignore" });
      timer = setTimeout(() => {
        child.unref();
        done(false);
      }, SPAWN_TIMEOUT_MS);
      child.once("error", () => {
        done(false);
      });
      child.once("spawn", () => {
        started = true;
        child.unref();
        // Started is not opened: give it a moment to fail.
        clearTimeout(timer);
        timer = setTimeout(() => {
          done(true);
        }, EARLY_EXIT_MS);
      });
      child.once("exit", (code) => {
        if (started) done(code === 0);
      });
    } catch {
      done(false);
    }
  });
}
