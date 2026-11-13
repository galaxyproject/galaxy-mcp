import type { Command } from "commander";
import { Option } from "commander";
import type { AuthDeps } from "./types";
import { loginCommand, type LoginOptions } from "./commands/login";
import { statusCommand, type StatusOptions } from "./commands/status";
import { tokenCommand, type TokenOptions } from "./commands/token";
import { envCommand, SHELLS, type EnvOptions } from "./commands/env";
import { headersCommand, type HeadersOptions } from "./commands/headers";
import { logoutCommand, type LogoutOptions } from "./commands/logout";
import { EX_SOFTWARE } from "../exit";
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
  // Without this it would be dropped without a word, by someone who thinks it was used.
  // `--profile` is the same for `login` alone: the other subcommands use it to find the host (and
  // `status` to say which key an operation would send), but a login never takes its host from a
  // profile, so there it would be dropped just as silently.
  auth.hook("preAction", (_auth, action) => {
    const g = program.opts<{ apiKey?: string; profile?: string; quiet?: boolean }>();
    if (g.quiet) return;
    // `status` reports on `--api-key` (what operations would use), so it says nothing of it here.
    if (g.apiKey && action.name() !== "status") {
      process.stderr.write(`Warning: --api-key is ignored by \`${commandName} auth\` commands.\n`);
    }
    if (g.profile && action.name() === "login") {
      process.stderr.write(
        `Warning: --profile is ignored by \`${commandName} auth login\`: it only uses --host or GALAXY_URL from the environment.\n`,
      );
    }
  });
  const host = () =>
    new Option("--host <url>", "Galaxy base URL (default: GALAXY_URL, or the only stored login)");

  const run = async (fn: (deps: AuthDeps) => Promise<number>): Promise<void> => {
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
          return await loginCommand({ ...o, signal: ac.signal }, deps);
        } finally {
          process.removeListener("SIGINT", onSig);
        }
      }),
    );

  auth
    .command("status")
    .description("Show stored logins and check them against Galaxy")
    .addOption(host())
    .option(
      "--json",
      "print JSON to stdout; exits 0 once the list is printed (each entry's status says how its check went)",
    )
    .option("--show-token", "do not mask tokens")
    .action((o: StatusOptions) =>
      run(async (deps) => {
        const ac = new AbortController();
        const onSig = () => {
          ac.abort();
        };
        process.once("SIGINT", onSig);
        try {
          return await statusCommand({ ...o, signal: ac.signal }, deps);
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
    .command("env")
    .description("Print GALAXY_URL and GALAXY_API_KEY assignments, for eval")
    .addOption(host())
    .addOption(new Option("--shell <shell>", "shell syntax").choices([...SHELLS]).default("sh"))
    .action((o: EnvOptions) => run((deps) => envCommand(o, deps)));

  auth
    .command("headers")
    .description("Print the request headers as JSON, for an MCP headersHelper")
    .addOption(host())
    .action((o: HeadersOptions) => run((deps) => headersCommand(o, deps)));

  auth
    .command("logout")
    .description("Delete the stored login; a pasted API key is not revoked")
    .addOption(host())
    .action((o: LogoutOptions) => run((deps) => logoutCommand(o, deps)));
}
