import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parse as parseYaml } from "yaml";
import { normalizeBaseUrl } from "./auth/base-url";
import { NoHostError, pickHost, type HostSource } from "./auth/host";
import { createCredentialStore } from "./auth/store";
import { sanitize } from "./auth/sanitize";
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
  /** Set when the credential store could not be read (corrupt file, keyring locked). */
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

/**
 * A named profile that is not in `~/.planemo.yml` is an error, not "no profile": it would
 * otherwise contribute neither a URL nor a key, and the run would quietly go to a stored login for
 * a different Galaxy than the one the person named. (An unreadable `~/.planemo.yml` reads as no
 * profiles, so it ends here too.) It is a usage error: exit 64.
 */
export function requireProfile(flags: { profile?: string }, src: { profiles: Record<string, unknown> }): void {
  const name = flags.profile;
  if (name && !Object.prototype.hasOwnProperty.call(src.profiles, name)) {
    throw new AuthError("usage", `Unknown --profile ${sanitize(name)}: no such profile in ~/.planemo.yml`);
  }
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

/** A Galaxy chosen to talk to, and the part of the rule that chose it. */
interface PickedHost { url: string; source: HostSource }

/** Why no key can be used. The data is what the message for it needs. */
export type KeyRefusal =
  /** The host rule refused (several logins stored and none named, or the like): its own error. */
  | { reason: "host"; error: unknown }
  /** Nothing to use; `named` is the Galaxy that was named, if any. */
  | { reason: "no_credentials"; named: string | undefined }
  /** `--profile` has a key but no galaxy_url, and no Galaxy is named. */
  | { reason: "profile_no_url" }
  /** `--profile` has a key but no galaxy_url, and the only Galaxy named is from `.env`. */
  | { reason: "profile_dotenv_host" }
  /** The stored login has expired and no other key can be used; `expiry` says how. */
  | { reason: "login_expired"; baseUrl: BaseUrl; expiry: string }
  /** The stored login is the one wanted but its token cannot be read (only with `unreadableStored`). */
  | { reason: "stored_unreadable" };

/** `picked` is the Galaxy the rule chose, or undefined when nothing was named and nothing is stored. */
export type KeyChoice =
  | { ok: true; picked: PickedHost | undefined; baseUrl: string; apiKey: string; keySource: KeySourceId; expired?: { baseUrl: BaseUrl; text: string } }
  | ({ ok: false; picked: PickedHost | undefined } & KeyRefusal);

/**
 * The decision behind `resolveConnection`, without the wording: which Galaxy and which key an
 * operation would use, or why it would use none. Pure; sources are injected.
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
 *
 * `opts.unreadableStored` says a login is stored for that Galaxy but its token cannot be read
 * (`src.storedLogin` is then absent). It counts as absent for the keys below it, and when nothing
 * else could be used the refusal is `stored_unreadable`, not "no credentials".
 *
 * It does not check `--profile` exists: `requireProfile` is the caller's to run first.
 */
export function chooseKey(flags: GlobalFlags, src: Sources, opts: { unreadableStored?: BaseUrl } = {}): KeyChoice {
  // The same rule as `gxy auth *` (auth/host.ts). With nothing named and nothing stored the
  // "no credentials" refusal says what to do, so that case is not a refusal of its own.
  let picked: PickedHost | undefined;
  try {
    picked = pickOperationHost(flags, src);
  } catch (e) {
    if (!(e instanceof NoHostError)) return { ok: false, picked: undefined, reason: "host", error: e };
  }
  const named = picked && picked.source !== "stored" ? picked.url : undefined;
  const dotenvKey = src.dotenv.GALAXY_API_KEY;
  // A key that did not come from a stored login goes to a named Galaxy or nowhere. A profile's key
  // goes to the profile's own URL, or to a host named in the command or the environment (`--url X
  // --profile p` asks for both: p's key to X is what was typed), never to one a `.env` chose.
  const sendTo = (apiKey: string, keySource: KeySourceId, expired?: { baseUrl: BaseUrl; text: string }): KeyChoice => {
    if (!named) {
      return keySource === "profile"
        ? { ok: false, picked, reason: "profile_no_url" }
        : { ok: false, picked, reason: "no_credentials", named };
    }
    if (keySource === "profile" && picked?.source === "dotenv") return { ok: false, picked, reason: "profile_dotenv_host" };
    return { ok: true, picked, baseUrl: named, apiKey, keySource, ...(expired ? { expired } : {}) };
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
    return { ok: true, picked, baseUrl: login.baseUrl, apiKey: login.token, keySource: "stored" };
  }
  if (login) {
    // The only other key that can be used is one for a host somebody named.
    if (dotenvKey && named) {
      return sendTo(dotenvKey, "dotenv", { baseUrl: login.baseUrl, text: expiredText(login, now) });
    }
    return { ok: false, picked, reason: "login_expired", baseUrl: login.baseUrl, expiry: expiredText(login, now) };
  }
  const rest = dotenvKey ? sendTo(dotenvKey, "dotenv") : { ok: false as const, picked, reason: "no_credentials" as const, named };
  if (!rest.ok && picked && opts.unreadableStored && isGalaxy(picked.url, opts.unreadableStored)) {
    return { ok: false, picked, reason: "stored_unreadable" };
  }
  return rest;
}

const expiredText = (login: StoredLogin, now: Date) => `Your login to ${login.baseUrl} ${describeExpiry(login.expiresAt!, now)}`;

/** The error for a Galaxy that was named, or none, that no key can be used for. */
function noCredentialsError(named: string | undefined, src: Sources, commandName: string): Error {
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
}

/**
 * Turn `chooseKey`'s answer into a connection, or the error for its refusal. `commandName` is only
 * for the hints in messages (`gxy`, or `galaxy-cli` for someone who typed that). See `chooseKey`
 * for which Galaxy and key are chosen.
 */
export function resolveConnection(flags: GlobalFlags, src: Sources, commandName = "gxy"): Connection {
  requireProfile(flags, src);
  const choice = chooseKey(flags, src);
  const dotenvFile = join(src.dotenvDir ?? ".", ".env");
  if (!choice.ok) {
    switch (choice.reason) {
      case "host":
        throw choice.error;
      case "no_credentials":
        throw noCredentialsError(choice.named, src, commandName);
      case "stored_unreadable": // only asked for by `keySourceFor`
        throw noCredentialsError(undefined, src, commandName);
      case "profile_no_url":
        throw new Error(
          `--profile ${flags.profile ?? ""} has a key but no galaxy_url, and no Galaxy is named. ` +
            "Add galaxy_url to the profile, or pass --url.",
        );
      case "profile_dotenv_host":
        throw new Error(
          `--profile ${flags.profile ?? ""} has a key but no galaxy_url, and the Galaxy named is from ${dotenvFile}; ` +
            "not sending the profile's key there. Add galaxy_url to the profile, or pass --url.",
        );
      case "login_expired":
        throw new AuthError("not_logged_in", `${choice.expiry}; run ${commandName} auth login --host ${choice.baseUrl}`);
    }
  }
  // A `.env` comes with the directory you are in, often a clone of somebody else's repository, so
  // say when it chose the Galaxy or the key, and when a login of yours outranked its key.
  const { baseUrl, apiKey, keySource } = choice;
  // A usable key was found, but something the person should know about was not: the login that
  // ran out, or the store that could not be read.
  const warning = choice.expired
    ? `${choice.expired.text}; using another key instead. Run ${commandName} auth login --host ${choice.expired.baseUrl} to renew it.`
    : keySource === "dotenv"
      ? src.storeProblem
      : undefined;
  const conn: Connection = { baseUrl, apiKey, keySource, ...(warning ? { warning } : {}) };
  const used = [keySource === "dotenv" ? "GALAXY_API_KEY" : "", choice.picked?.source === "dotenv" ? "GALAXY_URL" : ""].filter(Boolean);
  const lines = [];
  if (used.length) lines.push(`Using ${used.join(" and ")} from ${dotenvFile}.`);
  if (keySource === "stored" && src.dotenv.GALAXY_API_KEY) {
    lines.push(`The stored login for ${baseUrl} ranks above GALAXY_API_KEY in ${dotenvFile}, so that key is not used.`);
  }
  return lines.length ? { ...conn, note: lines.join("\n") } : conn;
}

/**
 * What `auth status` says about the key for one Galaxy: the source an operation would take it from,
 * or why it names none (the operation would go to another Galaxy, or would not run).
 */
type KeySourceReport =
  | { id: KeySourceId }
  | { id: null; reason: "elsewhere"; galaxy: string }
  | { id: null; reason: "ambiguous" | "refused" | "unreadable" };

/**
 * Which source an operation run now, with these flags and this environment, would take its key
 * from for this Galaxy, given its stored login: what `auth status` shows. It reads `chooseKey`'s
 * decision, so it names a source only when the operation would go to this Galaxy. An expired
 * login still counts as "stored": nothing else would be used for it either, and the operation says
 * why it fails. `src.storedBaseUrls` is every stored login, since several of them make an operation
 * that names no Galaxy ambiguous.
 *
 * `login` is undefined when its token cannot be read (a locked keyring), which is how an operation
 * sees it too: it uses another key if it has one, and otherwise the report says "unreadable" where
 * the stored login would have been the key.
 */
export function keySourceFor(baseUrl: BaseUrl, flags: GlobalFlags, src: Sources, login: StoredLogin | undefined): KeySourceReport {
  const all: Sources = { ...src, storedBaseUrls: src.storedBaseUrls ?? [baseUrl] };
  const choice = chooseKey(flags, { ...all, ...(login ? { storedLogin: login } : {}) }, login ? {} : { unreadableStored: baseUrl });
  if (!choice.picked) return { id: null, reason: "ambiguous" };
  let normalised: string | undefined;
  try {
    normalised = normalizeBaseUrl(choice.picked.url);
  } catch {
    normalised = undefined;
  }
  if (normalised !== baseUrl) return { id: null, reason: "elsewhere", galaxy: choice.picked.url };
  // `resolveConnection` would stop at an unknown profile before anything else.
  if (flags.profile && !Object.prototype.hasOwnProperty.call(src.profiles, flags.profile)) return { id: null, reason: "refused" };
  if (choice.ok) return { id: choice.keySource };
  if (choice.reason === "login_expired") return { id: "stored" };
  if (choice.reason === "stored_unreadable") return { id: null, reason: "unreadable" };
  return { id: null, reason: "refused" };
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
    case "unreadable":
      return "none (the stored token cannot be read)";
  }
}

/** The stored login, if it is for this very Galaxy: compared as normalised, never as typed. */
function matchingLogin(baseUrl: string, login: StoredLogin | undefined): StoredLogin | undefined {
  return login && isGalaxy(baseUrl, login.baseUrl) ? login : undefined;
}

/** Whether `url`, as typed, is this stored (normalised) base URL. */
function isGalaxy(url: string, stored: BaseUrl): boolean {
  try {
    return normalizeBaseUrl(url) === stored;
  } catch {
    return false;
  }
}

/** Read the real sources from disk/env (kept separate from the pure resolver). */
export function loadSources(): Sources {
  return { env: process.env, dotenv: readDotenv(), dotenvDir: process.cwd(), profiles: readPlanemoProfiles() };
}

/**
 * Like `loadSources`, plus the stored login for the Galaxy these flags point at. The store
 * is only touched when no key outranking it is given, so a broken keyring or hosts.yml never
 * gets in the way of `--api-key`, `--profile` or `GALAXY_API_KEY`.
 */
export async function loadSourcesAsync(
  flags: GlobalFlags = {},
  opts: { store?: CredentialStore; now?: Date; base?: Sources } = {},
): Promise<Sources> {
  const src = opts.base ?? loadSources();
  const now = opts.now ?? new Date();
  requireProfile(flags, src);
  if (flags.apiKey || profileKeyOf(flags, src) || src.env.GALAXY_API_KEY) return { ...src, now };
  try {
    // An operation is run by a person at a terminal as often as by a script: with stderr on a
    // terminal, a locked keyring may be unlocked at its prompt, as for `auth` (auth/deps.ts).
    const store =
      opts.store ??
      createCredentialStore(src.env, undefined, undefined, { interactive: process.stderr.isTTY === true });
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
