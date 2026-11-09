import { randomBytes } from "node:crypto";
import { constants } from "node:fs";
import {
  chmod,
  lstat,
  mkdir,
  open,
  readdir,
  rename,
  stat,
  unlink,
  type FileHandle,
} from "node:fs/promises";
import { dirname, join } from "node:path";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import { normalizeBaseUrl } from "./base-url";
import { sanitize } from "./sanitize";
import { AuthError, type BaseUrl, type LoginSummary } from "./types";

const STALE_TEMP_MS = 60 * 60 * 1000;

/** One host's entry in `hosts.yml`. `token` is present only when `storage` is `file`. */
export interface HostRecord extends LoginSummary {
  token?: string;
}

/**
 * What a read of `hosts.yml` yields: one entry per host, in file order, and any other top-level
 * keys. The rule for what this build cannot understand, in one place:
 *
 * - An entry this build does not recognise (a `storage` or `method` value it has never heard of,
 *   as a newer or older gxy might write, or a key that is not a normalised Galaxy URL) is
 *   *skipped* on a read, with a warning, so one such entry cannot lock out the others.
 * - It is never *dropped*: a write carries it back out unchanged, as it does unknown fields on
 *   other hosts' entries that we do read and unknown top-level keys. `put` and `delete` for that
 *   very host refuse instead of replacing it, because that would discard a login (and maybe a
 *   keyring secret) we could not read. (`put` for a host we can read writes its entry afresh, so
 *   unknown fields on that one entry are replaced.)
 * - A file with nothing we can place at all is *corrupt*, and every read and write refuses it: not
 *   a mapping, `hosts` not a mapping, a non-empty mapping with no `hosts` key (a misspelt key, or
 *   another layout), or a recognised entry with a bad field. Reading any of those as empty would
 *   let the next `put` overwrite the file and lose what is in it.
 *
 * Maps, not objects, so a host key of `__proto__` is an ordinary key and not the prototype.
 */
type Entry =
  | { kind: "known"; record: HostRecord; extra: Record<string, unknown> }
  | { kind: "unknown"; raw: unknown; why: string };
interface Doc {
  hosts: Map<string, Entry>;
  extra: Map<string, unknown>;
}

/** Keys of a host entry that this build reads or deliberately drops (`token` for keyring storage). */
const KNOWN_FIELDS = new Set([
  "username",
  "email",
  "method",
  "scopes",
  "expires_at",
  "obtained_at",
  "storage",
  "token",
]);

/**
 * `hosts.yml`: per-host login metadata, plus the token itself when the system keyring is
 * not used. The name, the keying by host and the plaintext fallback are gh's; the key here is
 * the whole base URL (see `normalizeBaseUrl`) because a Galaxy can live under a path prefix.
 *
 * The directory is 0700 and the file 0600, enforced on every write (and repaired, with a
 * warning, on a read too when an existing one is looser), because the file can hold a bearer token that
 * anyone able to read it could use as the user. A file that others can *write* is refused, not
 * repaired: narrowing its mode would not undo what they may already have put in it (a login with
 * their token for a real Galaxy URL, so that uploads go to their account). Writes go to a temp
 * file in the same directory (a rename is only atomic within one filesystem), are fsynced, then
 * renamed over the target, so a crash never leaves a half file.
 *
 * Before every read or write the directory and `hosts.yml` are `lstat`ed and refused if they
 * are symlinks or belong to another user: a link planted by someone else in a shared or
 * misconfigured home could redirect our write (and a later chmod) to a file of their choosing,
 * or make us trust content they wrote. The temp file is created exclusively (`wx`, so a
 * planted file or link at that name makes `open` fail rather than being written through) with
 * mode 0600 and then `fchmod`ed, because the umask can mask the creation mode but not the
 * chmod: under a restrictive umask the file would otherwise get a mode narrower than 0600,
 * and the rename carries the temp file's mode over.
 *
 * Read-modify-write is not locked across processes: two simultaneous logins can lose one
 * update. There are no refresh tokens, so this file is only written at login and logout, which
 * are rare and interactive; that is why it is accepted. (gh serialises login, logout and
 * refresh with a lock file.)
 */
export class FileStore {
  /** Entries already warned about, so a command that reads twice warns once. */
  private readonly warned = new Set<string>();

  constructor(
    readonly dir: string,
    readonly path: string = join(dir, "hosts.yml"),
    private readonly warn: (message: string) => void = () => {},
    /** Who must own the directory and file; undefined (Windows) skips the check. Injectable for tests. */
    private readonly uid: number | undefined = process.getuid?.(),
  ) {}

  async get(baseUrl: BaseUrl): Promise<HostRecord | null> {
    const entry = (await this.readAll()).hosts.get(baseUrl);
    if (entry?.kind === "unknown") throw this.unreadable(baseUrl, entry);
    return entry?.record ?? null;
  }

  async list(): Promise<HostRecord[]> {
    const { hosts } = await this.readAll();
    return [...hosts.keys()].sort().flatMap((k) => {
      const entry = hosts.get(k);
      return entry?.kind === "known" ? [entry.record] : [];
    });
  }

  async put(record: HostRecord): Promise<void> {
    const doc = await this.readAll();
    const old = doc.hosts.get(record.baseUrl);
    if (old?.kind === "unknown") throw this.unreadable(record.baseUrl, old);
    doc.hosts.set(record.baseUrl, { kind: "known", record, extra: {} });
    await this.writeAll(doc);
  }

  async delete(baseUrl: BaseUrl): Promise<boolean> {
    const doc = await this.readAll();
    const old = doc.hosts.get(baseUrl);
    if (!old) return false;
    if (old.kind === "unknown") throw this.unreadable(baseUrl, old);
    doc.hosts.delete(baseUrl);
    await this.writeAll(doc);
    return true;
  }

  /**
   * What `get`, `put` and `delete` say about a host whose entry this build skips: it says whose
   * login this is and what can be done, so a command for that host never reads as "not logged in".
   */
  private unreadable(url: string, entry: { why: string }): Error {
    return new AuthError(
      "store",
      `The login for ${show(url)} in ${this.path} was written by a newer version of gxy (${entry.why}), so this version leaves it as it is. Use that version, or delete the entry from the file and log in again with this one.`,
    );
  }

  /** Warn once per entry and file, however many times a command reads the store. */
  private warnSkipped(url: string, why: string): void {
    const key = `${this.path}\0${url}`;
    if (this.warned.has(key)) return;
    this.warned.add(key);
    this.warn(
      `${this.path}: skipping the entry for ${show(url)} (${why}); it is kept in the file as it is.`,
    );
  }

  private async readAll(): Promise<Doc> {
    await this.assertSafe();
    await this.tightenDir(false);
    let text: string;
    try {
      // One open, and every check below on this handle: a path-based `stat` or `chmod` would
      // follow a symlink swapped in after the `lstat` above. O_NOFOLLOW closes that gap for the
      // open itself.
      const fh = await open(this.path, constants.O_RDONLY | constants.O_NOFOLLOW);
      try {
        // Mode first, content second: once the file is 0600 nobody else can change it between
        // here and the parse.
        await this.checkOpenFile(fh);
        text = await fh.readFile("utf8");
      } finally {
        await fh.close();
      }
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT")
        return { hosts: new Map(), extra: new Map() };
      throw e;
    }
    if (!text.trim()) return { hosts: new Map(), extra: new Map() };
    // A file that cannot be understood is an error, never an empty list: `put` starts from this
    // read, so reading it as empty would overwrite it and lose every other login in it.
    let doc: unknown;
    try {
      doc = parseYaml(text);
    } catch (e) {
      // Not the parser's message: it quotes the offending source line, and in this file that
      // line can be the token. Only where the fault is.
      const pos = (e as { linePos?: { line: number; col: number }[] }).linePos?.[0];
      throw this.corrupt(
        pos ? `YAML syntax error at line ${pos.line}, column ${pos.col}` : "YAML syntax error",
      );
    }
    if (doc === null || doc === undefined) return { hosts: new Map(), extra: new Map() };
    if (!isObject(doc)) throw this.corrupt("expected a mapping");
    const top = Object.entries(doc);
    // An empty mapping (`{}`) is empty; a mapping with other keys and no `hosts` is not ours.
    if (!("hosts" in doc)) {
      if (top.length === 0) return { hosts: new Map(), extra: new Map() };
      throw this.corrupt("no `hosts` key");
    }
    const raw = doc.hosts;
    if (raw !== null && raw !== undefined && !isObject(raw))
      throw this.corrupt("`hosts` is not a mapping");
    const out: Doc = {
      hosts: new Map(),
      extra: new Map(top.filter(([k]) => k !== "hosts")),
    };
    for (const [url, value] of Object.entries(raw ?? {})) {
      const entry = this.decode(url, value);
      if (entry.kind === "unknown") this.warnSkipped(url, entry.why);
      out.hosts.set(url, entry);
    }
    return out;
  }

  /**
   * The file we opened is the one `assertSafe` looked at, ours and plain, and not writable by
   * anyone else. Group/other-readable is repaired (read bits only, on this handle) with a
   * warning, since reading is all that was exposed; group/other-*writable* is refused, since
   * someone else may have changed what is in it. POSIX only: there is no uid to compare on
   * Windows, and its modes mean nothing.
   */
  private async checkOpenFile(fh: FileHandle): Promise<void> {
    if (this.uid === undefined) return;
    const st = await fh.stat();
    const refuse = (why: string) =>
      new AuthError(
        "store",
        `Refusing to use ${this.path}: it ${why}. It would hold your login tokens, so it must be a plain file you own that only you can write.`,
      );
    if (!st.isFile()) throw refuse("is not a regular file");
    if (st.uid !== this.uid) throw refuse("belongs to another user");
    const mode = st.mode & 0o777;
    if (mode & 0o022)
      throw refuse(
        `is writable by others (mode ${mode.toString(8)}), so someone else may have changed it; check it, then run chmod 600 on it`,
      );
    if (mode & 0o077) {
      try {
        await fh.chmod(0o600);
        this.warn(`${this.path} was readable by others (${mode.toString(8)}); changed it to 600.`);
      } catch {
        // Not ours to fix (read-only mount): reading still works.
      }
    }
  }

  /**
   * Check every field rather than cast: a hand-edited or truncated file must fail naming the
   * entry, not surface later as a `TypeError`, or as a file login with no token. What is not
   * recognised at all becomes an `unknown` entry (see `Doc`); what is recognised but wrong is
   * corrupt. An `expires_at` that is a string but not a date is let through here and judged by
   * `checkExpiry`, which treats it as expired.
   */
  private decode(url: string, raw: unknown): Entry {
    if (!isObject(raw)) throw this.corrupt(`entry for ${show(url)} is not a mapping`);
    let normal = false;
    try {
      normal = normalizeBaseUrl(url) === url;
    } catch {
      // Not a URL at all: skipped below, like any other key that is not ours.
    }
    // A key that is not a normalised URL is not one we could have written, and is never looked up.
    // That also means a hand-edited `http://evil.example` key cannot be trusted with a token.
    if (!normal) return { kind: "unknown", raw, why: "not a normalised Galaxy URL" };
    // The two enumerations a later (or earlier) gxy may extend. Checked before anything else, so
    // an entry from another version is skipped whole rather than judged by our field rules. The
    // value itself is never echoed.
    const word = (k: string): string | undefined =>
      typeof raw[k] === "string" && raw[k] ? raw[k] : undefined;
    const storageWord = word("storage");
    if (storageWord !== undefined && storageWord !== "keyring" && storageWord !== "file")
      return { kind: "unknown", raw, why: "unknown storage" };
    const methodWord = word("method");
    if (methodWord !== undefined && methodWord !== "pasted")
      return { kind: "unknown", raw, why: "unknown method" };
    const str = (k: string): string => {
      const v = raw[k];
      if (typeof v !== "string" || !v) throw this.corrupt(`entry for ${show(url)} has no \`${k}\``);
      return v;
    };
    const storage = str("storage") as "keyring" | "file";
    const method = str("method") as "pasted";
    const scopes = raw.scopes ?? [];
    if (!Array.isArray(scopes) || scopes.some((s) => typeof s !== "string")) {
      throw this.corrupt(`entry for ${show(url)} has invalid scopes`);
    }
    const expires = raw.expires_at;
    if (expires !== null && expires !== undefined && typeof expires !== "string") {
      throw this.corrupt(`entry for ${show(url)} has invalid expires_at`);
    }
    const rec: HostRecord = {
      baseUrl: url as BaseUrl,
      username: str("username"),
      method,
      scopes: scopes as string[],
      expiresAt: expires ?? null,
      obtainedAt: str("obtained_at"),
      storage,
    };
    if (typeof raw.email === "string") rec.email = raw.email;
    if (storage === "file") rec.token = str("token");
    const extra = Object.fromEntries(Object.entries(raw).filter(([k]) => !KNOWN_FIELDS.has(k)));
    return { kind: "known", record: rec, extra };
  }

  /** The messages built here carry the path and a position, never file content: it holds tokens. */
  private corrupt(why: string): Error {
    return new AuthError(
      "store",
      `${this.path} is not valid (${why}). Fix or delete it and try again; it has not been modified.`,
    );
  }

  /**
   * The token is written only for `storage: file`, so a record that points at the keyring can
   * never leave a stale plaintext copy behind when a login moves there.
   */
  private encode(doc: Doc): string {
    const hosts = new Map<string, unknown>();
    for (const [url, entry] of doc.hosts) {
      if (entry.kind === "unknown") {
        hosts.set(url, entry.raw);
        continue;
      }
      const r = entry.record;
      hosts.set(url, {
        username: r.username,
        ...(r.email !== undefined ? { email: r.email } : {}),
        method: r.method,
        scopes: r.scopes,
        expires_at: r.expiresAt,
        obtained_at: r.obtainedAt,
        storage: r.storage,
        ...(r.storage === "file" && r.token !== undefined ? { token: r.token } : {}),
        ...entry.extra,
      });
    }
    return stringifyYaml(new Map<string, unknown>([["hosts", hosts], ...doc.extra]));
  }

  private async writeAll(doc: Doc): Promise<void> {
    await this.assertSafe();
    await mkdir(this.dir, { recursive: true, mode: 0o700 });
    // Again, because the first pass skips what does not exist yet: this one checks the
    // directory `mkdir` just made, or found.
    await this.assertSafe();
    await this.tightenDir(true);
    await this.removeStaleTemps();
    const tmp = join(
      dirname(this.path),
      `.hosts.${process.pid}.${randomBytes(6).toString("hex")}.tmp`,
    );
    // 'wx' + 0600 so the secret is never briefly readable under a looser umask. The pid and the
    // random part keep two writers off each other's temp file; `removeStaleTemps` matches the name.
    const fh = await open(tmp, "wx", 0o600);
    try {
      await fh.writeFile(this.encode(doc), "utf8");
      await fh.chmod(0o600);
      await fh.sync();
    } catch (e) {
      await fh.close().catch(() => {});
      await unlink(tmp).catch(() => {});
      throw e;
    }
    await fh.close();
    try {
      await rename(tmp, this.path);
    } catch (e) {
      await unlink(tmp).catch(() => {});
      throw e;
    }
    // Make the rename itself durable. Best effort: not every platform can open a directory.
    try {
      const dh = await open(this.dir, "r");
      try {
        await dh.sync();
      } finally {
        await dh.close();
      }
    } catch {
      // Ignored on purpose.
    }
  }

  /**
   * Narrow a config directory that others can enter to 0700, with a warning, as is done for the
   * file: on a read as well as a write, since the file's mode already protects the tokens and the
   * directory only decides who can list it. A failure to narrow it is not fatal on a read (a
   * read-only mount: reading still works), but is on a write. POSIX only, like the file's mode.
   */
  private async tightenDir(forWrite: boolean): Promise<void> {
    if (this.uid === undefined) return;
    let mode: number;
    try {
      mode = (await stat(this.dir)).mode & 0o777;
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return;
      throw e;
    }
    if (!(mode & 0o077)) return;
    try {
      await chmod(this.dir, 0o700);
    } catch (e) {
      if (forWrite) throw e;
      return;
    }
    this.warn(`${this.dir} was accessible to others (${mode.toString(8)}); changed it to 700.`);
  }

  /**
   * Refuse a config directory or `hosts.yml` that is a symlink, the wrong kind, or someone
   * else's. Only the last component of each path is looked at, so a symlinked parent of the
   * directory is accepted. A loose mode is repaired elsewhere with a warning; ownership is
   * refused here, because a file that belongs to another user is not ours to fix.
   */
  private async assertSafe(): Promise<void> {
    for (const [what, p, kind] of [
      ["directory", this.dir, "dir"],
      ["file", this.path, "file"],
    ] as const) {
      let st;
      try {
        st = await lstat(p);
      } catch (e) {
        if ((e as NodeJS.ErrnoException).code === "ENOENT") continue;
        throw e;
      }
      const problem = st.isSymbolicLink()
        ? "is a symbolic link"
        : (kind === "dir" ? !st.isDirectory() : !st.isFile())
          ? `is not a regular ${what}`
          : this.uid !== undefined && st.uid !== this.uid
            ? "belongs to another user"
            : undefined;
      if (problem) {
        throw new AuthError(
          "store",
          `Refusing to use ${p}: it ${problem}. It would hold your login tokens, so it must be a plain ${what} you own. Remove it or point GXY_CONFIG_DIR somewhere else.`,
        );
      }
    }
  }

  /**
   * A write killed between creating its temp file and the rename leaves `.hosts.*.tmp`, which
   * may hold a token. Remove the ones old enough that no live write can own them. Best effort.
   */
  private async removeStaleTemps(): Promise<void> {
    try {
      const cutoff = Date.now() - STALE_TEMP_MS;
      for (const name of await readdir(this.dir)) {
        if (!/^\.hosts\..+\.tmp$/.test(name)) continue;
        const p = join(this.dir, name);
        const st = await lstat(p);
        if (st.isFile() && st.mtimeMs < cutoff) await unlink(p);
      }
    } catch {
      // Cleaning up is never worth failing a login for.
    }
  }
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** File-supplied text for a message: no control characters (they could drive the terminal), bounded. */
function show(text: string): string {
  const clean = sanitize(text);
  return clean.length > 120 ? `${clean.slice(0, 120)}...` : clean;
}
