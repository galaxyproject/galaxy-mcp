import type { Command } from "commander";
import { Option } from "commander";
import type { AuthDeps } from "./types";
import { loginCommand, type LoginOptions } from "./commands/login";
import { statusCommand, type StatusOptions } from "./commands/status";
import { tokenCommand, type TokenOptions } from "./commands/token";
import { envCommand, SHELLS, type EnvOptions } from "./commands/env";
import { headersCommand, type HeadersOptions } from "./commands/headers";
import { logoutCommand, type LogoutOptions } from "./commands/logout";
import { EX_SOFTWARE, EX_USAGE } from "../exit";
import { exitQuietlyOnEpipe } from "./epipe";

/**
 * `--scopes a,b`, `--scopes "a b"` or `--scopes a --scopes b`: every use adds to the list,
 * in order, without repeats. Commander hands each use the list so far; a parser that ignored
 * `previous` would keep only the last `--scopes`, and a login would ask for less than was typed.
 */
export function parseScopes(value: string, previous: string[] = []): string[] {
  const added = value.split(/[,\s]+/).filter(Boolean);
  return [...new Set([...previous, ...added])];
}

/**
 * Add the `auth` group. `makeDeps` is called per invocation, so `--help` and the other
 * commands never touch the keyring or the network.
 */
export function registerAuth(program: Command, makeDeps: (commandName: string) => AuthDeps): void {
  const commandName = program.name();
  const auth = program
    .command("auth")
    .description("Log in to a Galaxy and manage stored credentials");
  // `--api-key` belongs to the operations; `auth` takes its key from the store or from stdin.
  // `--profile` is the same for `login` and `headers`: the other subcommands use it to find the
  // host (and `status` to say which key an operation would send), but a login never takes its host
  // from a profile, and `headers` takes it only from the server Claude Code names. `status` also reports on `--api-key` (what operations would use), so it is the one
  // command that takes it. A flag that does not apply is a usage error
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
    if (g.profile !== undefined && action.name() === "headers") {
      problems.push(
        `--profile does not apply to \`${name}\`: it takes its Galaxy from CLAUDE_CODE_MCP_SERVER_URL.`,
      );
    }
    if (g.apiKey !== undefined && action.name() !== "status") {
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
  // Login takes a new key, so it never falls back to a stored login (see pickHost) and says so.
  const host = (fallback = "GALAXY_URL, or the only stored login") =>
    new Option("--host <url>", `Galaxy base URL (default: ${fallback})`);

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
    .description("Log in with the device flow, or store an API key from stdin")
    .addOption(host("GALAXY_URL, or asked for at a terminal"))
    .option(
      "--scopes <list>",
      "scopes to request, comma- or space-separated (default: the server's)",
      parseScopes,
    )
    .option("--no-browser", "do not offer to open the approval page in a browser")
    .option("--with-token", "read an existing API key from stdin instead of the device flow")
    .option("--insecure-storage", "store the token in a file instead of the system keyring")
    .action((o: LoginOptions) =>
      run(async (deps) => {
        // Ctrl-C aborts one signal that every step of the login takes (discovery, the code
        // request, the prompts, the poll), so it ends with "Cancelled." and 130. A code or token
        // the server issued at that very moment is taken back by `login` itself with its own
        // bounded call; one issued too late to see expires on its own. `once`, so a second
        // Ctrl-C falls through to Node's default and ends the process.
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
    .addOption(
      // Checked by `envCommand`, so a bad value is a usage error (64) like the other commands'.
      new Option("--shell <shell>", `shell syntax: ${SHELLS.join(", ")}`).default("sh"),
    )
    .action((o: EnvOptions) => run((deps) => envCommand(o, deps)));

  auth
    .command("headers")
    .description("Print the request headers as JSON, for an MCP headersHelper")
    .addOption(host("the Galaxy CLAUDE_CODE_MCP_SERVER_URL names, which it must match"))
    .action((o: HeadersOptions) => run((deps) => headersCommand(o, deps)));

  auth
    .command("logout")
    .description("Revoke the token where possible and delete the stored login")
    .addOption(host())
    .action((o: LogoutOptions) => run((deps) => logoutCommand(o, deps)));
}
