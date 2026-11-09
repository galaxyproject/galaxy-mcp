import { Command } from "commander";
import {
  allOperations,
  describeOperation,
  runWithEnvelope,
  createGalaxyContext,
  type GalaxyContext,
} from "@galaxyproject/galaxy-ops";
import { resolveConnection, loadSourcesAsync, type Connection } from "./config";
import { applyInputs } from "./flags-apply";
import { buildInput, inCliNames, keptPositionals } from "./flags";
import { render, type Format } from "./render";
import { exitCodeFor, exitCodeForAuthError, EX_USAGE, EX_SOFTWARE } from "./exit";
import { registerAuth } from "./auth/register";
import { createAuthDeps } from "./auth/deps";
import { AuthError, type AuthDeps } from "./auth/types";

export interface CliDeps {
  makeContext?: (conn: Connection, signal: AbortSignal) => GalaxyContext;
  /** Tests inject fakes here; by default the real wiring in `auth/deps.ts`. */
  makeAuthDeps?: (commandName: string) => AuthDeps;
  /**
   * The name the program prints in usage and hints: `gxy` or `galaxy-cli`. Each name has its own
   * entry file (`gxy.ts`, `index.ts`) that passes it, because the name a person typed is not
   * available at run time: pnpm and Windows shims run `node dist/<file>.js`, so `process.argv[1]`
   * is the file and never the command.
   */
  invokedAs?: string;
}

export function buildProgram(deps: CliDeps = {}): Command {
  const program = new Command();
  program
    .name(deps.invokedAs ?? "galaxy-cli")
    .description("Galaxy agent operations on the command line")
    .option("--url <url>", "Galaxy base URL")
    .option("--api-key <key>", "Galaxy API key (deprecated: use `auth login` or GALAXY_API_KEY)")
    .option("--profile <name>", "planemo profile name")
    .option("--format <fmt>", "output format: table|json|text", "table")
    .option("--quiet", "suppress status messages", false)
    .option("--timeout <ms>", "poll timeout for blocking ops");

  // `auth` shares the operations' host rule, so it needs the global --url/--profile (read at run time).
  registerAuth(
    program,
    deps.makeAuthDeps ??
      ((name) => createAuthDeps(name, () => program.opts<{ url?: string; profile?: string }>())),
  );

  for (const op of allOperations) {
    const cmd = program.command(op.name).description(inCliNames(describeOperation(op), op.input));
    applyInputs(cmd, op);
    cmd.action(async (...args: unknown[]) => {
      // commander passes positionals..., the command's own options, then the Command.
      const command = args[args.length - 1] as Command;
      const localOpts = args[args.length - 2] as Record<string, unknown>;
      const positionals = args.slice(0, -2) as (string | undefined)[];
      const globals = command.optsWithGlobals<{ url?: string; apiKey?: string; profile?: string; format: Format; quiet: boolean; timeout?: string }>();

      // Still works, so a script that passes it keeps running; stderr, so `--format json` output
      // stays parseable, and `--quiet` silences it for a wrapper that fails on any stderr.
      if (globals.apiKey && !globals.quiet) {
        console.error(
          `Warning: --api-key is deprecated: the key is visible in shell history and process listings. Use \`${program.name()} auth login\` or set GALAXY_API_KEY instead.`,
        );
      }

      const parsed = buildInput(op.input, positionals, localOpts, keptPositionals(op.name));
      if (!parsed.success) {
        console.error(parsed.error.message);
        process.exitCode = EX_USAGE;
        return;
      }
      // When a context factory is injected (tests), skip credential resolution entirely.
      let conn: Connection = { baseUrl: "", apiKey: "", keySource: "flag" };
      if (!deps.makeContext) {
        try {
          const flags = { url: globals.url, apiKey: globals.apiKey, profile: globals.profile };
          conn = resolveConnection(flags, await loadSourcesAsync(flags), program.name());
          // A key was found but something is off (an unreadable store, an expired login with
          // another key used instead): worth a line, not a failure.
          if (conn.warning) console.error(`Warning: ${conn.warning}`);
          if (conn.note && !globals.quiet) console.error(conn.note);
        } catch (e) {
          console.error((e as Error).message);
          // An expired login is 77 and an ambiguous host 64, as for `auth`; anything else is a usage problem.
          process.exitCode = e instanceof AuthError ? exitCodeForAuthError(e.kind) : EX_USAGE;
          return;
        }
      }

      const ac = new AbortController();
      const onSig = () => ac.abort();
      process.once("SIGINT", onSig);
      const ctx = (deps.makeContext ?? ((c, s) => createGalaxyContext({ ...c, signal: s, poll: globals.timeout ? { timeoutMs: Number(globals.timeout) } : undefined })))(conn, ac.signal);
      try {
        // No serializer of its own: the budget is measured on the compact line,
        // as it is on the MCP surface and on the Python server, and the indented
        // printing happens afterwards. A page is the same page whoever reads it.
        const result = await runWithEnvelope(op as never, parsed.data as never, ctx);
        render(result, { format: globals.format, quiet: globals.quiet });
        process.exitCode = exitCodeFor(result.errorKind);
      } catch (e) {
        console.error((e as Error).stack ?? String(e)); // non-Galaxy error = a bug
        process.exitCode = EX_SOFTWARE;
      } finally {
        process.removeListener("SIGINT", onSig);
      }
    });
  }
  return program;
}
