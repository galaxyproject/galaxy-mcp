import { homedir } from "node:os";
import { isAbsolute, join, win32 } from "node:path";

/**
 * Where `gxy` keeps its files: `$GXY_CONFIG_DIR`, else `$XDG_CONFIG_HOME/gxy`, else
 * `~/.config/gxy`: gh's order (`GH_CONFIG_DIR`, `XDG_CONFIG_HOME`, `~/.config/gh`) with `gxy` in
 * place of `gh`. Env and home directory are parameters so tests never see the real ones.
 *
 * Empty variables count as unset, and a relative `GXY_CONFIG_DIR` or `XDG_CONFIG_HOME` is ignored
 * (the XDG spec says so for its own): a relative path would put the credential file wherever the
 * command happened to be run, possibly inside a repository. `isAbsolute` rather than a leading
 * `/`, so a Windows path counts too.
 */
export function configDir(env: Record<string, string | undefined>, home: string): string {
  const own = env.GXY_CONFIG_DIR;
  if (own && isAbsolute(own)) return own;
  const xdg = env.XDG_CONFIG_HOME;
  if (xdg && isAbsolute(xdg)) return join(xdg, "gxy");
  return join(home, ".config", "gxy");
}

/**
 * What to say about who can read the plaintext `hosts.yml` in `dir`, as a phrase for "(...)".
 *
 * On POSIX the file is 0600 in a 0700 directory, so "readable only by you" is true. On Windows
 * mode bits do nothing: the file gets whatever its folder's ACL passes down. Under the user's
 * profile that is the user, SYSTEM and Administrators, so the claim holds. Anywhere else (a
 * `GXY_CONFIG_DIR` on a shared drive, `C:\tools`) the folder typically passes down
 * `BUILTIN\Users` read access, so the claim would be false and we say what is true instead.
 * Platform and home are parameters so tests need no Windows.
 */
export function fileAccessPhrase(
  dir: string,
  platform: NodeJS.Platform = process.platform,
  home: string = homedir(),
): string {
  if (platform !== "win32") return "readable only by you";
  const rel = win32.relative(home, dir);
  const inProfile = rel === "" || (!rel.startsWith("..") && !win32.isAbsolute(rel));
  return inProfile
    ? "readable only by you"
    : "with the permissions it inherits from its folder, so other users may be able to read it";
}
