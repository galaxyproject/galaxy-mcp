import type { Command } from "commander";
import { Option } from "commander";
import type { AuthDeps } from "./types";
import { loginCommand, type LoginOptions } from "./commands/login";
import { tokenCommand, type TokenOptions } from "./commands/token";
import { logoutCommand, type LogoutOptions } from "./commands/logout";
import { EX_SOFTWARE, EX_USAGE } from "../exit";
import { exitQuietlyOnEpipe } from "./epipe";

/**
 * Add the `auth` group. `makeDeps` is called per invocation, so `--help` and the other
 * commands never touch the store or the network.
 */
export function registerAuth(program: Command, makeDeps: (commandName: string) => AuthDeps): void {
  const commandName = program.name();
  const auth = program
    .command("auth")
    .description("Log in to a Galaxy and manage stored credentials");
  // `--api-key` belongs to the operations; `auth` takes its key from the store or from stdin.
  // `--profile` is the same for `login` alone: the other subcommands use it to find the host, but
  // a login never takes its host from a profile. A flag that does not apply is a usage error
  // (64), as in `gh` and `aws`, and not a warning: `login --profile prod --with-token` would
  // otherwise send the key piped on stdin to a Galaxy other than prod's before anyone could
  // react. `--quiet` does not silence it. The hook runs before the action, so nothing is read,
  // asked, stored or sent; `run` then sees `refused` and does nothing.
  let refused = false;
  auth.hook("preAction", (_auth, action) => {
    refused = false;
    const g = program.opts<{ apiKey?: string; profile?: string }>();
    const name = `${commandName} auth ${action.name()}`;
    const problems: string[] = [];
    if (g.profile !== undefined && action.name() === "login") {
      problems.push(
        `--profile does not apply to \`${name}\`: login takes its Galaxy from --host, --url or GALAXY_URL in the environment.`,
      );
    }
    if (g.apiKey !== undefined) {
      problems.push(
        action.name() === "login"
          ? `--api-key does not apply to \`${name}\`: pipe the key in with --with-token instead.`
          : `--api-key does not apply to \`${name}\`.`,
      );
    }
    if (problems.length === 0) return;
    refused = true;
    process.stderr.write(`${problems.join("\n")}\n`);
    process.exitCode = EX_USAGE;
  });
  const host = () =>
    new Option("--host <url>", "Galaxy base URL (default: GALAXY_URL, or the only stored login)");

  const run = async (fn: (deps: AuthDeps) => Promise<number>): Promise<void> => {
    if (refused) return;
    exitQuietlyOnEpipe();
    let deps: AuthDeps;
    try {
      deps = makeDeps(commandName);
    } catch (e) {
      // Building the wiring failed before anything the person typed was looked at, so this is
      // the CLI's fault: 70 (EX_SOFTWARE), not a usage error.
      process.stderr.write(`${(e as Error).message}\n`);
      process.exitCode = EX_SOFTWARE;
      return;
    }
    process.exitCode = await fn(deps);
  };

  auth
    .command("login")
    .description("Store an API key read from stdin")
    .addOption(host())
    .option("--with-token", "read the API key from stdin")
    .option(
      "--insecure-storage",
      "keep an admin key in the plaintext file (it is refused otherwise)",
    )
    .action((o: LoginOptions) =>
      run(async (deps) => {
        // Ctrl-C aborts one signal that every step of the login takes (the probe, the prompt, the
        // whoami call), so it ends with "Cancelled." and 130. `once`, so a second Ctrl-C falls
        // through to Node's default and ends the process.
        const ac = new AbortController();
        const onSig = () => {
          ac.abort();
        };
        process.once("SIGINT", onSig);
        try {
          return await loginCommand(
            { ...o, signal: ac.signal, quiet: program.opts<{ quiet?: boolean }>().quiet },
            deps,
          );
        } finally {
          process.removeListener("SIGINT", onSig);
        }
      }),
    );

  auth
    .command("token")
    .description("Print the stored token (stdout)")
    .addOption(host())
    .option("--json", "print a JSON object with the token and its expiry")
    .action((o: TokenOptions) => run((deps) => tokenCommand(o, deps)));

  auth
    .command("logout")
    .description("Delete the stored login; a pasted API key is not revoked")
    .addOption(host())
    .action((o: LogoutOptions) => run((deps) => logoutCommand(o, deps)));
}
