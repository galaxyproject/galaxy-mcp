import { Command } from "commander";
import {
  allOperations,
  runWithEnvelope,
  createGalaxyContext,
  type GalaxyContext,
} from "@galaxyproject/galaxy-ops";
import { resolveConnection, loadSources, type Connection } from "./config";
import { applyInputs } from "./flags-apply";
import { buildInput } from "./flags";
import { render, type Format } from "./render";
import { exitCodeFor, EX_USAGE, EX_SOFTWARE } from "./exit";

export interface CliDeps {
  makeContext?: (conn: Connection, signal: AbortSignal) => GalaxyContext;
}

export function buildProgram(deps: CliDeps = {}): Command {
  const program = new Command();
  program
    .name("galaxy-cli")
    .description("Galaxy agent operations on the command line")
    .option("--url <url>", "Galaxy base URL")
    .option("--api-key <key>", "Galaxy API key")
    .option("--profile <name>", "planemo profile name")
    .option("--format <fmt>", "output format: table|json|text", "table")
    .option("--quiet", "suppress status messages", false)
    .option("--timeout <ms>", "poll timeout for blocking ops");

  for (const op of allOperations) {
    const cmd = program.command(op.name).description(op.summary);
    applyInputs(cmd, op);
    cmd.action(async (...args: unknown[]) => {
      // commander passes positionals..., the command's own options, then the Command.
      const command = args[args.length - 1] as Command;
      const localOpts = args[args.length - 2] as Record<string, unknown>;
      const positionals = args.slice(0, -2) as string[];
      const globals = command.optsWithGlobals<{ url?: string; apiKey?: string; profile?: string; format: Format; quiet: boolean; timeout?: string }>();

      const parsed = buildInput(op.input, positionals, localOpts);
      if (!parsed.success) {
        console.error(parsed.error.message);
        process.exitCode = EX_USAGE;
        return;
      }
      // When a context factory is injected (tests), skip credential resolution entirely.
      let conn: Connection = { baseUrl: "", apiKey: "" };
      if (!deps.makeContext) {
        try {
          conn = resolveConnection({ url: globals.url, apiKey: globals.apiKey, profile: globals.profile }, loadSources());
        } catch (e) {
          console.error((e as Error).message);
          process.exitCode = EX_USAGE;
          return;
        }
      }

      const ac = new AbortController();
      const onSig = () => ac.abort();
      process.once("SIGINT", onSig);
      const ctx = (deps.makeContext ?? ((c, s) => createGalaxyContext({ ...c, signal: s, poll: globals.timeout ? { timeoutMs: Number(globals.timeout) } : undefined })))(conn, ac.signal);
      try {
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
