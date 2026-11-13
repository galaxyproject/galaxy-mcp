import { AuthError, type AuthDeps } from "../types";
import { fail, requireLogin } from "./common";

type Shell = "sh" | "fish" | "powershell";
export const SHELLS: readonly Shell[] = ["sh", "fish", "powershell"];

export interface EnvOptions {
  host?: string;
  shell?: string;
}

/** Single-quote for the shell, so no value can break out of the quotes. */
export function quoteFor(shell: Shell, value: string): string {
  switch (shell) {
    case "sh":
      return `'${value.replace(/'/g, `'\\''`)}'`;
    case "fish":
      // Inside fish single quotes only \' and \\ are special.
      return `'${value.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
    case "powershell":
      // PowerShell also treats the curly quotes U+2018 to U+201B as single quotes
      // (about_Quoting_Rules), and doubles any of them the same way.
      return `'${value.replace(/['\u2018-\u201b]/g, (q) => q + q)}'`;
  }
}

function assign(shell: Shell, name: string, value: string): string {
  const q = quoteFor(shell, value);
  switch (shell) {
    case "sh":
      return `export ${name}=${q}`;
    case "fish":
      return `set -gx ${name} ${q}`;
    case "powershell":
      return `$env:${name} = ${q}`;
  }
}

/**
 * Print `GALAXY_URL` and `GALAXY_API_KEY`, the two variables galaxy-cli's operations and the
 * Python galaxy-mcp server both read, so one `eval "$(gxy auth env)"` points every Galaxy tool in
 * that shell at this login.
 *
 * It also puts the token in the environment of every process started from that shell, which is
 * why the README tells agents to run `gxy <operation>` instead. An exported `GALAXY_API_KEY` is
 * used by operations before any stored login (config.ts), so it will not follow a later `login`.
 */
export async function envCommand(opts: EnvOptions, deps: AuthDeps): Promise<number> {
  try {
    const shell = (opts.shell ?? "sh") as Shell;
    if (!SHELLS.includes(shell))
      throw new AuthError("usage", `Unknown shell "${shell}". Use one of: ${SHELLS.join(", ")}.`);
    const login = await requireLogin(deps, opts.host);
    deps.io.stdout.write(
      `${assign(shell, "GALAXY_URL", login.baseUrl)}\n${assign(shell, "GALAXY_API_KEY", login.token)}\n`,
    );
    return 0;
  } catch (e) {
    return fail(deps, e);
  }
}
