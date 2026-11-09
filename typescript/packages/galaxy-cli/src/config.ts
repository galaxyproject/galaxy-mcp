import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parse as parseYaml } from "yaml";
import { normalizeBaseUrl } from "./auth/base-url";
import { NoHostError, pickHost, type HostSource } from "./auth/host";
import { createCredentialStore } from "./auth/store";
import { describeExpiry, isExpired } from "./auth/expiry";
import { AuthError, type BaseUrl, type CredentialStore, type KeySourceId, type StoredLogin } from "./auth/types";

/** `warning` is set when a usable key was found but something the user should know about was not. */
export interface Connection {
  baseUrl: string;
  apiKey: string;
  /** Where the key came from. */
  keySource: KeySourceId;
  warning?: string;
  /** Worth saying unless `--quiet`: a `.env` was used, or a login outranked its key. */
  note?: string;
}
export interface GlobalFlags { url?: string; apiKey?: string; profile?: string; }
interface PlanemoProfile { galaxy_url?: string; galaxy_user_key?: string; galaxy_admin_key?: string; }
export interface Sources {
  env: Record<string, string | undefined>;
  dotenv: Record<string, string | undefined>;
  profiles: Record<string, PlanemoProfile>;
  /** Base URLs of every stored login (`gxy auth login`), loaded before resolving. */
  storedBaseUrls?: BaseUrl[];
  /** The stored login for the Galaxy being used (or the only one), with its token. */
  storedLogin?: StoredLogin;
  /** The clock, for judging whether `storedLogin` has expired. */
  now?: Date;
  /** The directory `dotenv` was read from, for saying so. */
  dotenvDir?: string;
  /** Set when the credential store could not be read (corrupt file). */
  storeProblem?: string;
}

/**
 * Which Galaxy an operation talks to: `auth/host.ts`'s one rule, fed from these sources.
 *
 * The `--profile` URL ranks second: a profile is a pair, and its key must go to its own Galaxy and
 * not to whatever `GALAXY_URL` or a `.env` says. `auth` (except `login`) uses the same order.
 *
 * `forLogin` is left off. An operation has always read `.env` and the profile, and a stored token
 * only goes to a host that matches a stored login, so those files cannot send one anywhere; it is
 * `auth login` that must not take its host from them.
 */
function pickOperationHost(flags: GlobalFlags, src: Sources): { url: string; source: HostSource } {
  const prof = flags.profile ? src.profiles[flags.profile] : undefined;
  return pickHost({
    flag: flags.url,
    env: src.env,
    dotenv: src.dotenv,
    profileUrl: prof?.galaxy_url,
    storedBaseUrls: src.storedBaseUrls ?? [],
  });
}

/** What `--api-key`, `--profile`, `GALAXY_API_KEY`, a stored login and `.env` are called in output. */
export function describeKeySource(id: KeySourceId, flags: GlobalFlags = {}): string {
  switch (id) {
    case "flag":
      return "--api-key";
    case "profile":
      return `--profile ${flags.profile ?? ""}`.trim();
    case "env":
      return "GALAXY_API_KEY";
    case "stored":
      return "stored login";
    case "dotenv":
      return ".env";
  }
}

/** The key of the `--profile` planemo profile, if one is named and has one. */
function profileKeyOf(flags: GlobalFlags, src: Sources): string | undefined {
  const prof = flags.profile ? src.profiles[flags.profile] : undefined;
  return prof?.galaxy_user_key ?? prof?.galaxy_admin_key;
}

/**
 * Pure resolver -- sources injected so it is fully testable. `commandName` is only for the hints
 * in messages (`gxy`, or `galaxy-cli` for someone who typed that).
 *
 * The key, first match: `--api-key`; the `--profile` key; `GALAXY_API_KEY` in the environment;
 * the stored login for the host; `.env`'s key. The command line beats the environment, which
 * beats files, as in the aws CLI: what you typed for this run is never overridden by what was left
 * lying around, and a `.env` in a directory you happen to be in is the weakest claim of all. Before
 * stored logins existed `--profile` ranked below the environment and `.env`; it was the one
 * command-line choice that lost to files.
 *
 * Only a stored login's own token may go to the host that the stored logins supplied (the only
 * one, when nothing names a host). Any other key goes only to a host somebody named: otherwise a
 * `.env` for some other Galaxy would send its key to wherever you last logged in. The Galaxy is
 * chosen the same way: `--url`, the `--profile` URL, `GALAXY_URL`, `.env`'s, the only stored login.
 */
export function resolveConnection(flags: GlobalFlags, src: Sources, commandName = "gxy"): Connection {
  // The same rule as `gxy auth *` (auth/host.ts). With nothing named and nothing stored the
  // generic "no credentials" error below says what to do, so that case is not an error here.
  let picked: { url: string; source: HostSource } | undefined;
  try {
    picked = pickOperationHost(flags, src);
  } catch (e) {
    if (!(e instanceof NoHostError)) throw e;
  }
  const named = picked && picked.source !== "stored" ? picked.url : undefined;
  const noCredentials = () => {
    const problem = src.storeProblem ? ` (${src.storeProblem})` : "";
    // A Galaxy was named, and logins are stored, but none of them is for it: say so, rather than the
    // generic advice, which reads as if a login existed and were not found. Shown as a URL without
    // anything a user could have put in it (its credentials, its query): normalised, else the host.
    if (named && src.storedBaseUrls?.length) {
      let key: string | undefined;
      let shown: string;
      try {
        key = normalizeBaseUrl(named);
        shown = key;
      } catch {
        try {
          shown = new URL(named).host;
        } catch {
          shown = "the Galaxy named";
        }
      }
      if (!key || !src.storedBaseUrls.includes(key as BaseUrl)) {
        const logins = src.storedBaseUrls.join(", ");
        return new Error(
          `No Galaxy credentials for ${shown}: the stored login${src.storedBaseUrls.length > 1 ? "s are" : " is"} for ${logins}. ` +
            `Run ${commandName} auth login --host ${shown}, or set GALAXY_API_KEY (env, a .env file, or --api-key) for it.${problem}`,
        );
      }
    }
    return new Error(
      "No Galaxy credentials. Set GALAXY_URL and GALAXY_API_KEY (env, a .env file, or --url/--api-key), " +
        `or pass --profile <name> for a ~/.planemo.yml profile, or run ${commandName} auth login.${problem}`,
    );
  };
  const dotenvFile = join(src.dotenvDir ?? ".", ".env");
  const dotenvKey = src.dotenv.GALAXY_API_KEY;
  // A `.env` comes with the directory you are in, often a clone of somebody else's repository, so
  // say when it chose the Galaxy or the key, and when a login of yours outranked its key.
  const finish = (conn: Connection): Connection => {
    const used = [
      conn.keySource === "dotenv" ? "GALAXY_API_KEY" : "",
      picked?.source === "dotenv" ? "GALAXY_URL" : "",
    ].filter(Boolean);
    const lines = [];
    if (used.length) lines.push(`Using ${used.join(" and ")} from ${dotenvFile}.`);
    if (conn.keySource === "stored" && dotenvKey) {
      lines.push(
        `The stored login for ${conn.baseUrl} ranks above GALAXY_API_KEY in ${dotenvFile}, so that key is not used.`,
      );
    }
    return lines.length ? { ...conn, note: lines.join("\n") } : conn;
  };
  // A key that did not come from a stored login goes to a named Galaxy or nowhere. A profile's key
  // goes to the profile's own URL, or to a host named in the command or the environment (`--url X
  // --profile p` asks for both: p's key to X is what was typed), never to one a `.env` chose.
  const sendTo = (apiKey: string, keySource: KeySourceId, warning?: string): Connection => {
    if (!named) {
      if (keySource === "profile") {
        throw new Error(
          `--profile ${flags.profile ?? ""} has a key but no galaxy_url, and no Galaxy is named. ` +
            "Add galaxy_url to the profile, or pass --url.",
        );
      }
      throw noCredentials();
    }
    if (keySource === "profile" && picked?.source === "dotenv") {
      throw new Error(
        `--profile ${flags.profile ?? ""} has a key but no galaxy_url, and the Galaxy named is from ${dotenvFile}; ` +
          "not sending the profile's key there. Add galaxy_url to the profile, or pass --url.",
      );
    }
    return finish({ baseUrl: named, apiKey, keySource, ...(warning ? { warning } : {}) });
  };

  if (flags.apiKey) return sendTo(flags.apiKey, "flag");
  const profileKey = profileKeyOf(flags, src);
  if (profileKey) return sendTo(profileKey, "profile");
  if (src.env.GALAXY_API_KEY) return sendTo(src.env.GALAXY_API_KEY, "env");

  const login = picked ? matchingLogin(picked.url, src.storedLogin) : undefined;
  const now = src.now ?? new Date();
  // An unreadable expiry counts as expired, and so does one within a minute of running out.
  if (login && !isExpired(login.expiresAt, now)) {
    // The key goes to the vetted form of the URL it was stored for, not to the raw string
    // (which may carry a query or an odd path) that merely normalises to it.
    return finish({ baseUrl: login.baseUrl, apiKey: login.token, keySource: "stored" });
  }
  if (login) {
    const expired = `Your login to ${login.baseUrl} ${describeExpiry(login.expiresAt!, now)}`;
    // The only other key that can be used is one for a host somebody named.
    if (dotenvKey && named) {
      return sendTo(dotenvKey, "dotenv", `${expired}; using another key instead. Run ${commandName} auth login --host ${login.baseUrl} to renew it.`);
    }
    throw new AuthError("not_logged_in", `${expired}; run ${commandName} auth login --host ${login.baseUrl}`);
  }
  if (dotenvKey) return sendTo(dotenvKey, "dotenv", src.storeProblem);
  throw noCredentials();
}

/**
 * What `auth status` says about the key for one Galaxy: the source an operation would take it from,
 * or why it names none (the operation would go to another Galaxy, or would not run).
 */
type KeySourceReport =
  | { id: KeySourceId }
  | { id: null; reason: "elsewhere"; galaxy: string }
  | { id: null; reason: "ambiguous" | "refused" };

/**
 * Which source an operation run now, with these flags and this environment, would take its key
 * from for this Galaxy, given its stored login (if any): what `auth status` shows. It runs the real
 * resolver, so it names a source only when the operation would go to this Galaxy. An expired
 * login still counts as "stored": nothing else would be used for it either, and the operation says
 * why it fails. `src.storedBaseUrls` is every stored login, since several of them make an operation
 * that names no Galaxy ambiguous.
 */
export function keySourceFor(baseUrl: BaseUrl, flags: GlobalFlags, src: Sources, login?: StoredLogin): KeySourceReport {
  const all: Sources = { ...src, storedBaseUrls: src.storedBaseUrls ?? [baseUrl], ...(login ? { storedLogin: login } : {}) };
  let picked: { url: string };
  try {
    picked = pickOperationHost(flags, all);
  } catch {
    return { id: null, reason: "ambiguous" };
  }
  let normalised: string | undefined;
  try {
    normalised = normalizeBaseUrl(picked.url);
  } catch {
    normalised = undefined;
  }
  if (normalised !== baseUrl) return { id: null, reason: "elsewhere", galaxy: picked.url };
  try {
    return { id: resolveConnection(flags, all).keySource };
  } catch (e) {
    return e instanceof AuthError && e.kind === "not_logged_in" ? { id: "stored" } : { id: null, reason: "refused" };
  }
}

/** `describeKeySource`, or for a Galaxy with no source named the reason, as `auth status` words it. */
export function describeKeyReport(report: KeySourceReport, flags: GlobalFlags = {}): string {
  if (report.id) return describeKeySource(report.id, flags);
  switch (report.reason) {
    case "elsewhere": {
      let host = "another Galaxy";
      try {
        host = new URL(report.galaxy).host;
      } catch {
        // Not a URL: operations would fail on it, but all that is shown here is that it is not this one.
      }
      return `none (operations go to ${host} with these flags)`;
    }
    case "ambiguous":
      return "none (several logins are stored, and operations need --url to say which)";
    case "refused":
      return "none (operations with these flags stop with an error)";
  }
}

/** The stored login, if it is for this very Galaxy: compared as normalised, never as typed. */
function matchingLogin(baseUrl: string, login: StoredLogin | undefined): StoredLogin | undefined {
  if (!login) return undefined;
  try {
    return normalizeBaseUrl(baseUrl) === login.baseUrl ? login : undefined;
  } catch {
    return undefined;
  }
}

/** Read the real sources from disk/env (kept separate from the pure resolver). */
export function loadSources(): Sources {
  return { env: process.env, dotenv: readDotenv(), dotenvDir: process.cwd(), profiles: readPlanemoProfiles() };
}

/**
 * Like `loadSources`, plus the stored login for the Galaxy these flags point at. The store
 * is only touched when no key outranking it is given, so a broken hosts.yml never
 * gets in the way of `--api-key`, `--profile` or `GALAXY_API_KEY`.
 */
export async function loadSourcesAsync(
  flags: GlobalFlags = {},
  opts: { store?: CredentialStore; now?: Date; base?: Sources } = {},
): Promise<Sources> {
  const src = opts.base ?? loadSources();
  const now = opts.now ?? new Date();
  if (flags.apiKey || profileKeyOf(flags, src) || src.env.GALAXY_API_KEY) return { ...src, now };
  try {
    const store = opts.store ?? createCredentialStore(src.env);
    const summaries = await store.list();
    const storedBaseUrls = summaries.map((s) => s.baseUrl);
    let target: BaseUrl | undefined;
    try {
      // Only a host that matches a stored login exactly gets its token: a `.env` or profile
      // that names another Galaxy finds no login here, so nothing is sent there.
      target = normalizeBaseUrl(pickOperationHost(flags, { ...src, storedBaseUrls }).url);
    } catch (e) {
      // Ambiguity is reported by resolveConnection; an unparsable URL has no stored login.
      if (!(e instanceof AuthError)) throw e;
      target = undefined;
    }
    // Asked of the store for any named Galaxy, not only a listed one: the list leaves out a login this
    // version cannot read (one a newer gxy wrote), and `get` is what says so (as the store problem) instead of
    // leaving a plain "no credentials". For a Galaxy with no login at all it returns null: no token is read.
    const storedLogin = target ? ((await store.get(target)) ?? undefined) : undefined;
    return { ...src, storedBaseUrls, ...(storedLogin ? { storedLogin } : {}), now };
  } catch (e) {
    return { ...src, now, storeProblem: e instanceof Error ? e.message : String(e) };
  }
}

function readDotenv(): Record<string, string | undefined> {
  try {
    const text = readFileSync(join(process.cwd(), ".env"), "utf8");
    const out: Record<string, string> = {};
    for (const line of text.split("\n")) {
      const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$/);
      if (m?.[1] !== undefined && m[2] !== undefined) out[m[1]] = m[2].replace(/^["']|["']$/g, "");
    }
    return out;
  } catch {
    return {};
  }
}

function readPlanemoProfiles(): Record<string, PlanemoProfile> {
  try {
    const doc = parseYaml(readFileSync(join(homedir(), ".planemo.yml"), "utf8")) as Record<string, unknown>;
    // planemo stores named profiles under `profiles:`; also accept top-level galaxy_url/key as "default".
    const profiles = (doc?.profiles as Record<string, PlanemoProfile>) ?? {};
    if (doc?.galaxy_url) profiles.default = { galaxy_url: String(doc.galaxy_url), galaxy_user_key: doc.galaxy_user_key as string | undefined };
    return profiles;
  } catch {
    return {};
  }
}
