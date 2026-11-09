import { homedir } from "node:os";
import { FileStore } from "./file-store";
import { configDir } from "./paths";
import type { BaseUrl, CredentialStore, LoginSummary, StoredLogin, StorageKind } from "./types";

/**
 * Logins kept in `hosts.yml`, token and all: the file is readable only by its owner, as gh
 * does.
 */
export class FileCredentialStore implements CredentialStore {
  constructor(private readonly file: FileStore) {}

  async get(baseUrl: BaseUrl): Promise<(StoredLogin & { storage: StorageKind }) | null> {
    return this.file.get(baseUrl);
  }

  async put(login: StoredLogin): Promise<void> {
    await this.file.put({ ...login, storage: "file" });
  }

  delete(baseUrl: BaseUrl): Promise<boolean> {
    return this.file.delete(baseUrl);
  }

  async list(): Promise<LoginSummary[]> {
    return (await this.file.list()).map(({ token: _token, ...s }) => s);
  }
}

/**
 * The real store, the only one: config dir from the environment. One factory, so every auth
 * command agrees about where logins live. Warnings go to stderr, which is where everything
 * meant for a person goes.
 */
export function createCredentialStore(
  env: Record<string, string | undefined> = process.env,
  home: string = homedir(),
  warn: (message: string) => void = (m) => process.stderr.write(`warning: ${m}\n`),
): CredentialStore {
  return new FileCredentialStore(new FileStore(configDir(env, home), undefined, warn));
}
