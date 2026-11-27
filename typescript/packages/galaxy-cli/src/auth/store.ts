import { homedir } from "node:os";
import { abortReason } from "./abort";
import { FileStore, type HostRecord } from "./file-store";
import { KeyringStore, noKeyringBackend } from "./keyring-store";
import { configDir, fileAccessPhrase } from "./paths";
import { sanitize } from "./sanitize";
import {
  AuthError,
  type BaseUrl,
  type CredentialStore,
  type LoginSummary,
  type PutResult,
  type StoredLogin,
  type StorageKind,
} from "./types";

/**
 * Metadata always lives in `hosts.yml`; the token lives in the system keyring when that
 * works, else in `hosts.yml` too (with a warning). The metadata's `storage` field says
 * which, so `get` never has to guess, and a login kept in the file never touches the keyring.
 * It is the split gh makes, with the same plaintext fallback and an `--insecure-storage` of
 * the same name. What is added here: a keyring write must read back, and an admin token never
 * falls back to the file on its own.
 */
export class CombinedStore implements CredentialStore {
  constructor(
    private readonly file: FileStore,
    private readonly keyring: KeyringStore,
  ) {}

  async get(baseUrl: BaseUrl): Promise<(StoredLogin & { storage: StorageKind }) | null> {
    const rec = await this.file.get(baseUrl);
    if (!rec) return null;
    const { token: fileToken, ...summary } = rec;
    if (rec.storage === "file") return { ...summary, token: fileToken ?? "" };
    let token: string | null;
    try {
      token = await this.keyring.get(baseUrl, rec.username);
    } catch (e) {
      // `not_logged_in` (exit 77): the record exists but its token cannot be reached, which for
      // whoever is asking comes to the same thing as having no login.
      throw new AuthError(
        "not_logged_in",
        `The token for ${baseUrl} is in the system keyring, which is not available: ${(e as Error).message}`,
        { cause: e },
      );
    }
    // The record says the token is in the keyring and it is not there: that is a login we
    // cannot use, not "no login", and callers must not read it as a clean slate. A locked or
    // closed keyring can look exactly like a missing entry: keyring-rs unlocks items a search
    // finds, but KDE's ksecretd (KWallet) returns no match at all while the wallet is closed, so
    // the read comes back as "no entry" with nothing to tell the two apart. Hence both causes are
    // named, and `hosts.yml` is left as it is so that unlocking and retrying works.
    if (token === null) {
      throw new AuthError(
        "not_logged_in",
        `The login record for ${baseUrl} says its token is in the system keyring, but the keyring did not return it. It may be locked or closed (KDE Wallet looks empty while it is closed): unlock it and retry. If it is unlocked, the entry is gone: log in again to replace it. The login record has not been changed.`,
      );
    }
    return { ...summary, token };
  }

  /**
   * The old token is in a keyring that GXY_NO_KEYRING switches off: replacing the record would
   * leave that token where nothing points at it, when rerunning without the variable replaces
   * both. So nothing is stored, as for `delete`, whatever `--insecure-storage` says.
   */
  private refuseToOrphan(baseUrl: BaseUrl, prior: HostRecord | null): void {
    if (prior?.storage === "keyring" && this.keyring.disabled) {
      throw new AuthError(
        "usage",
        `The login for ${baseUrl} (user ${sanitize(prior.username)}) keeps its token in the system keyring, which GXY_NO_KEYRING switches off, so nothing was stored: replacing it now would leave that token behind. Rerun without GXY_NO_KEYRING.`,
      );
    }
  }

  async preflight(baseUrl: BaseUrl): Promise<void> {
    this.refuseToOrphan(baseUrl, await this.file.get(baseUrl));
  }

  async put(
    login: StoredLogin,
    opts: { insecureStorage?: boolean; requireKeyring?: boolean; signal?: AbortSignal } = {},
  ): Promise<PutResult> {
    const { token, ...summary } = login;
    // Read first: what the old record points at decides what must be cleaned up afterwards.
    const prior = await this.file.get(login.baseUrl);
    this.refuseToOrphan(login.baseUrl, prior);
    // The same user's earlier login is in the keyring: the write below replaces that very entry.
    const samePrior = prior?.storage === "keyring" && prior.username === login.username;
    // Read before overwriting, so a put that fails can put it back. Only on this path: a first
    // login, or one that moves from the file or to another user, has nothing to restore.
    // `undefined`: it could not be read. `null`: it was already missing, so nothing is lost.
    let oldSecret: string | null | undefined;
    if (samePrior && !opts.insecureStorage) {
      oldSecret = await this.keyring.get(login.baseUrl, login.username).catch(() => undefined);
    }
    const user = sanitize(login.username);
    // After a failed put whose keyring write landed: take the new secret back out, or, when the
    // entry is the one the existing record uses, put the old secret back. Best effort; returns a
    // note for the person when the keyring is left holding what no record accounts for.
    const undo = async (): Promise<string | undefined> => {
      try {
        if (!samePrior) {
          await this.keyring.delete(login.baseUrl, login.username);
          return undefined;
        }
        if (oldSecret === null) return undefined;
        if (oldSecret === undefined) throw new Error("it could not be read beforehand");
        await this.keyring.set(login.baseUrl, login.username, oldSecret);
        return undefined;
      } catch (e) {
        const why = sanitize(e instanceof Error ? e.message : String(e)).replace(/\.+$/, "");
        return samePrior
          ? `The earlier login for ${login.baseUrl} (user ${user}) may no longer work: its token in the system keyring was replaced and could not be put back (${why}). Log in again to fix it`
          : `The new token for ${login.baseUrl} (user ${user}) may still be in the system keyring, which gxy could not use to remove it (${why}). Remove the entry "gxy:${login.baseUrl}" for that user with your keyring's own tool`;
      }
    };
    let storage: StorageKind = "file";
    let warning: string | undefined;
    if (!opts.insecureStorage) {
      // Whether the keyring accepted the write: only then is there anything to take back out.
      let wrote = false;
      try {
        await this.keyring.set(login.baseUrl, login.username, token);
        wrote = true;
        // Read it back: a backend that accepts writes it cannot serve is no keyring at all.
        if ((await this.keyring.get(login.baseUrl, login.username)) !== token) {
          throw new Error("the token did not read back");
        }
        storage = "keyring";
      } catch (e) {
        // The write may have landed even though it did not read back (or the read threw). Take
        // it out, best effort, so no copy of the token lives outside the record that is about
        // to say "file"; when an existing record points at that very entry, put the old secret
        // back instead, since that record stays if this put is refused or cancelled. (When the
        // put falls back to the file, the old entry is removed below, as for any other change.)
        // A `set` that itself failed or timed out stored nothing we can see, and a delete after a
        // timeout would only wait for the same wedged keyring again, so it is skipped.
        let note: string | undefined;
        if (wrote && (!samePrior || opts.requireKeyring || opts.signal?.aborted)) {
          note = await undo();
        }
        // A Ctrl-C while the keyring was working (an unlock prompt can last a minute) kills its
        // helper with the process group, which looks like an unavailable keyring. It is a cancel:
        // no fallback to the file and no warning, so the caller's abort handling takes over.
        if (opts.signal?.aborted) throw abortReason(opts.signal);
        // Why, in the warning: a timeout, a locked store and a missing native module need
        // different fixes, and "no keyring available" for all of them leaves the person guessing.
        const why = sanitize(e instanceof Error ? e.message : String(e)).replace(/\.+$/, "");
        if (opts.requireKeyring) {
          // Nothing has been written to the file yet (it is only touched below), so refusing is clean.
          // `denied` is exit 77 (EX_NOPERM): the store declined by policy, nothing else broke. An
          // admin token is the one credential never left in plaintext as a side effect.
          throw new AuthError(
            "denied",
            `${this.keyring.disabled ? "The system keyring is switched off by GXY_NO_KEYRING" : `No system keyring here (${why})`}, so nothing was stored. Rerun with --insecure-storage to keep it in ${this.file.path} (${fileAccessPhrase(this.file.dir)}), or use GALAXY_API_KEY for this session.${note ? ` ${note}.` : ""}`,
            { cause: e },
          );
        }
        // Said aloud, never silent: a fallback nobody hears about is a plaintext token that the
        // user believes is in the keyring.
        warning = `${this.keyring.disabled ? "GXY_NO_KEYRING is set" : `No system keyring available (${why})`}; token saved in plain text at ${this.file.path} (${fileAccessPhrase(this.file.dir)})${note ? `. ${note}` : ""}`;
      }
    }
    const record: HostRecord =
      storage === "file" ? { ...summary, storage, token } : { ...summary, storage };
    // Metadata (and so the plaintext token, if any, being dropped) is replaced in one atomic write.
    try {
      await this.file.put(record);
    } catch (e) {
      // The secret is in the keyring but no record points at it: take it back out (or restore the
      // old one, when it overwrote the entry an existing record still points at), best effort.
      const note = storage === "keyring" ? await undo() : undefined;
      if (note && e instanceof Error) {
        // The failure itself is rethrown, carrying what the keyring was left holding.
        const msg = `${e.message}. ${note}`;
        throw e instanceof AuthError
          ? new AuthError(e.kind, msg, { cause: e })
          : new Error(msg, { cause: e });
      }
      throw e;
    }
    // Clean up what the new record no longer points at. The new login is stored either way, but a
    // secret that no record points at is one `logout` can never find, so when it cannot be removed
    // (the keyring is locked or not answering) the person is told it remains. (With the keyring
    // switched off by GXY_NO_KEYRING the put was refused above.)
    let leftover: string | undefined;
    if (prior?.storage === "keyring" && (storage === "file" || prior.username !== login.username)) {
      try {
        await this.keyring.delete(login.baseUrl, prior.username);
      } catch (e) {
        const why = sanitize(e instanceof Error ? e.message : String(e)).replace(/\.+$/, "");
        leftover = `The earlier token for ${login.baseUrl} (user ${sanitize(prior.username)}) may still be in the system keyring, which gxy could not use to remove it (${why}). Remove the entry "gxy:${login.baseUrl}" for that user with your keyring's own tool`;
      }
    }
    const warnings = [warning, leftover].filter((w): w is string => w !== undefined);
    return warnings.length ? { storage, warning: warnings.join(". ") } : { storage };
  }

  async delete(baseUrl: BaseUrl): Promise<boolean> {
    const rec = await this.file.get(baseUrl);
    if (!rec) return false;
    // With the keyring switched off rather than broken, removing the record alone would orphan a
    // token that nothing points at, when rerunning without GXY_NO_KEYRING would remove both. So
    // nothing is removed, and the person is told how to log out.
    if (rec.storage === "keyring" && this.keyring.disabled) {
      throw new AuthError(
        "usage",
        `The token for ${baseUrl} (user ${sanitize(rec.username)}) is in the system keyring, which GXY_NO_KEYRING switches off, so nothing was removed. Rerun without GXY_NO_KEYRING to log out.`,
      );
    }
    let leftover: unknown;
    if (rec.storage === "keyring") {
      try {
        const existed = await this.keyring.delete(baseUrl, rec.username);
        // The record says there is a secret and the keyring says there was nothing to delete. A
        // locked or closed keyring looks exactly like that (KDE's ksecretd returns no match while
        // the wallet is closed), and the record is then the only pointer to the secret. So it is
        // kept, and logout can be run again once the keyring is open; dropping it would leave a
        // token that nothing refers to and `logout` can never find.
        if (!existed) {
          throw new AuthError(
            "leftover",
            `The system keyring found no secret for ${baseUrl} (user ${sanitize(rec.username)}) to remove. It may be locked or closed, in which case the secret may remain there: unlock it and run logout again. The login record was kept for that. If the entry is really gone, delete the entry for ${baseUrl} from ${this.file.path} (or remove "gxy:${baseUrl}" for that user with your keyring's own tool)`,
          );
        }
      } catch (e) {
        if (e instanceof AuthError && e.kind === "leftover") throw e;
        leftover = e;
      }
    }
    // The local record goes either way: a login that cannot be found cannot be used by mistake.
    await this.file.delete(baseUrl);
    if (leftover) {
      throw new AuthError(
        "leftover",
        `The login record was removed, but the system keyring could not confirm that the secret for ${baseUrl} (user ${sanitize(rec.username)}) is gone, so it may remain there: ${(leftover as Error).message}. Remove the entry "gxy:${baseUrl}" for that user with your keyring's own tool`,
        { cause: leftover },
      );
    }
    return true;
  }

  /** Reads `hosts.yml` only, so `status` and host resolution still work when the keyring is wedged. */
  async list(): Promise<LoginSummary[]> {
    return (await this.file.list()).map(({ token: _token, ...s }) => s);
  }
}

/**
 * The real store, the only one: config dir from the environment, the native keyring unless
 * `GXY_NO_KEYRING` is set (to anything but "", "0" or "false"). One factory, so the auth
 * commands and operations (`config.ts`) cannot disagree about where logins live. Warnings go to
 * stderr, because stdout carries only what a pipe may read (a token).
 *
 * `interactive` says a person is at the terminal and may be asked to unlock the keyring, so
 * keyring calls wait a minute rather than 5 s and the person is told they are waiting. The
 * caller passes it (`auth` and operations each know their own terminal): the store never reads the TTY.
 * `notify` is where that notice goes, stderr by default.
 */
export function createCredentialStore(
  env: Record<string, string | undefined> = process.env,
  home: string = homedir(),
  warn: (message: string) => void = (m) => process.stderr.write(`warning: ${m}\n`),
  opts: { interactive?: boolean; notify?: (message: string) => void } = {},
): CredentialStore {
  const off =
    env.GXY_NO_KEYRING &&
    env.GXY_NO_KEYRING !== "0" &&
    env.GXY_NO_KEYRING.toLowerCase() !== "false";
  const notify = opts.notify ?? ((m: string) => process.stderr.write(`${m}\n`));
  return new CombinedStore(
    new FileStore(configDir(env, home), undefined, warn),
    new KeyringStore(off ? noKeyringBackend : undefined, undefined, {
      interactive: opts.interactive ?? false,
      notify,
    }),
  );
}
