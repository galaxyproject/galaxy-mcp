import { isAbsolute, join } from "node:path";

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
