import { describeKeyReport, keySourceFor, loadSources } from "../config";
import { resolveHost } from "./host";
import { createCredentialStore } from "./store";
import type { AuthDeps } from "./types";
import { createWhoami } from "./whoami";

/** `isTTY` is typed `boolean` but is `undefined` at runtime when the stream is a pipe or a file. */
const isTty = (stream: { isTTY?: boolean }): boolean => stream.isTTY === true;

/** The real wiring of `auth`, built once per invocation (so `--help` never touches any of it). */
export function createAuthDeps(
  commandName: string,
  globals: () => { url?: string; apiKey?: string; profile?: string } = () => ({}),
): AuthDeps {
  const env = process.env;
  const interactive = isTty(process.stdin) && isTty(process.stderr);
  const store = createCredentialStore(env);
  const now = () => new Date();
  return {
    store,
    whoami: createWhoami(),
    // The rule operations use (config.ts), so the two cannot pick different Galaxies. The global
    // `--url` counts as the flag and `--profile` supplies its planemo URL; login ignores `.env`
    // and the profile (see pickHost).
    resolveHost: (explicit, purpose) => {
      const g = globals();
      const src = loadSources();
      return resolveHost({
        explicit: explicit ?? g.url,
        env,
        dotenv: src.dotenv,
        profileUrl: g.profile ? src.profiles[g.profile]?.galaxy_url : undefined,
        forLogin: purpose === "login",
        store,
      });
    },
    // What an operation would use for this Galaxy, by the one order in config.ts. The global
    // `--url`, `--api-key` and `--profile` count: they are what `gxy <operation>` would be given.
    keySourceFor: (baseUrl, login, stored) => {
      const g = globals();
      const flags = { url: g.url, apiKey: g.apiKey, profile: g.profile };
      const src = { ...loadSources(), storedBaseUrls: [...stored], now: now() };
      const report = keySourceFor(baseUrl, flags, src, login);
      return { id: report.id, label: describeKeyReport(report, flags) };
    },
    ignoredLoginHost: () => {
      const src = loadSources();
      const dotenv = src.dotenv.GALAXY_URL;
      if (dotenv) return { source: "dotenv", url: dotenv };
      const profileName = globals().profile;
      const profile = profileName ? src.profiles[profileName]?.galaxy_url : undefined;
      return profile ? { source: "profile", url: profile } : undefined;
    },
    now,
    env,
    io: {
      stdin: process.stdin,
      stdout: process.stdout,
      stderr: process.stderr,
      // stdin and stderr, not stdout: every prompt goes to stderr and is answered on stdin, and
      // stdout is left free to be piped (`eval "$(gxy auth env)"`) without switching prompts off.
      // The key prompt of `login --with-token` looks at stdin alone, and refuses when stdin is a
      // terminal and this is false (stderr redirected), so the key never falls through to an echoing read (login.ts).
      interactive,
    },
    commandName,
  };
}
