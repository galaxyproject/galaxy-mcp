import { describe, it, expect, vi, beforeEach } from "vitest";
import { type ZodTypeAny } from "zod";
import { buildProgram } from "../src/program";
import { classifyField, inCliNames } from "../src/flags";
import { allOperations, createGalaxyContext } from "@galaxyproject/galaxy-ops";

function ctxFactory() {
  // a context whose client returns canned data for any GET
  return createGalaxyContext({
    baseUrl: "https://g.example",
    apiKey: "K",
    fetchImpl: (async () =>
      new Response(JSON.stringify([{ id: "h1", name: "alpha" }]), { status: 200, headers: { "content-type": "application/json" } })) as typeof fetch,
  });
}

/**
 * One CLI run, judged on its own. Each run gets its own console capture and its own exit status,
 * because a shared spy lets an earlier success answer for a later failure and the next test's
 * reset hides the exit code.
 */
async function runCli(argv: string[], makeContext: () => ReturnType<typeof createGalaxyContext>) {
  const out = vi.spyOn(console, "log").mockImplementation(() => {});
  const err = vi.spyOn(console, "error").mockImplementation(() => {});
  process.exitCode = 0;
  try {
    await buildProgram({ makeContext }).parseAsync(["node", "galaxy-cli", ...argv]);
    return {
      stdout: out.mock.calls.flat().join(""),
      stderr: err.mock.calls.flat().join(""),
      exitCode: process.exitCode,
    };
  } finally {
    out.mockRestore();
    err.mockRestore();
  }
}

/** Answers the invocation detail route with one record and the index with a list, recording each URL. */
function invocationsContext(asked: string[]) {
  return () =>
    createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
        asked.push(url);
        const body = url.includes("/api/invocations/")
          ? { id: "1e1d0c0b0a090807", state: "scheduled" }
          : [{ id: "1e1d0c0b0a090807", state: "scheduled" }];
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as typeof fetch,
    });
}

/**
 * Answers any GET with an empty list, recording every URL asked for. The version route
 * answers 26.1 so an op with a requirement gets past its guard and on to the call this
 * is about; `asked` therefore holds that request too.
 */
function recordingContext(asked: string[], body: unknown = []) {
  return () =>
    createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
        asked.push(url);
        const payload = url.includes("/api/version")
          ? { version_major: "26.1", version_minor: "26.1.1" }
          : body;
        return new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as typeof fetch,
    });
}

/** The one recorded URL for a given route, so a version lookup cannot be mistaken for it. */
const asking = (asked: string[], route: string): string => {
  const hit = asked.find((url) => url.includes(route));
  expect(hit, `nothing asked for ${route}`).toBeDefined();
  return hit!;
};

describe("buildProgram", () => {
  beforeEach(() => { process.exitCode = 0; });

  it("registers one subcommand per op", () => {
    const program = buildProgram({ makeContext: ctxFactory });
    const names = program.commands.map((c) => c.name());
    expect(names).toContain("get_histories");
    expect(names).toContain("create_history");
    expect(names.length).toBe(allOperations.length);
  });

  it("says in --help what an op needs from the server", () => {
    const program = buildProgram({ makeContext: ctxFactory });
    const described = new Map(program.commands.map((c) => [c.name(), c.description()]));
    for (const op of allOperations) {
      const text = described.get(op.name) ?? "";
      // The op's summary, with any parameter it names spelled the way a command line takes
      // it, and then the bound. The spelling itself is held to the two tests at the bottom
      // of this file, which derive it rather than agreeing with the code that applies it.
      expect(text.startsWith(inCliNames(op.summary, op.input)), op.name).toBe(true);
      expect(text.includes("Requires Galaxy"), op.name).toBe(op.requires !== undefined);
    }
    expect(described.get("list_page_revisions")).toContain("Requires Galaxy 26.1 or newer.");
    // get_page works on 26.0, so it must not pick the sentence up.
    expect(described.get("get_page")).not.toContain("Requires Galaxy");
  });

  it("takes get_invocations' id as a flag now that it is optional", async () => {
    // It used to be a required positional, so `galaxy-cli get_invocations abc123` worked. The
    // parity change made it optional, which moves it to --invocation-id.
    const asked: string[] = [];
    const run = await runCli(["get_invocations", "--invocation-id", "1e1d0c0b0a090807", "--format", "json"], invocationsContext(asked));
    expect(asked).toEqual([expect.stringContaining("/api/invocations/1e1d0c0b0a090807")]);
    expect(run.stdout).toContain('"success": true');
    expect(run.exitCode).toBe(0);
  });

  /**
   * The same move for get_job_details' dataset id, except that this one is kept: 0.3.1 is a
   * patch release, and `galaxy-cli get_job_details <id>` is what every script written against
   * 0.3.0 runs. So the id is a positional as before and `--dataset-id` like every other
   * optional, and giving it both ways is a usage error rather than one quietly winning.
   */
  it("still takes get_job_details' dataset id as a positional, and as --dataset-id", async () => {
    const job = { creating_job: "0123456789abcdef", id: "0123456789abcdef" };
    for (const argv of [["get_job_details", "d1"], ["get_job_details", "--dataset-id", "d1"]]) {
      const asked: string[] = [];
      const run = await runCli([...argv, "--format", "json"], recordingContext(asked, job));
      expect(asking(asked, "/api/datasets/d1"), argv.join(" ")).toBeDefined();
      expect(run.stdout, argv.join(" ")).toContain('"success": true');
      expect(run.exitCode, argv.join(" ")).toBe(0);
    }
  });

  it("refuses get_job_details' dataset id given both ways", async () => {
    const asked: string[] = [];
    const run = await runCli(
      ["get_job_details", "d1", "--dataset-id", "d2", "--format", "json"],
      recordingContext(asked),
    );
    expect(run.exitCode).toBe(64);
    expect(run.stderr).toContain("datasetId was given twice");
    expect(asked).toEqual([]);
  });

  it("takes get_job_details' job id as a flag, with the positional left out", async () => {
    const asked: string[] = [];
    const run = await runCli(
      ["get_job_details", "--job-id", "0123456789abcdef", "--format", "json"],
      recordingContext(asked, { id: "0123456789abcdef", state: "ok" }),
    );
    expect(asked).toEqual([expect.stringContaining("/api/jobs/0123456789abcdef")]);
    expect(run.stdout).toContain('"success": true');
    expect(run.exitCode).toBe(0);
  });

  it("lists invocations when get_invocations is given no id", async () => {
    const asked: string[] = [];
    const run = await runCli(["get_invocations", "--format", "json"], invocationsContext(asked));
    expect(asked).toEqual([expect.stringContaining("/api/invocations?")]);
    expect(asked[0]).not.toContain("workflow_id=");
    expect(run.stdout).toContain('"success": true');
    expect(run.exitCode).toBe(0);
  });

  /**
   * Every op whose default is now declared, driven through the real program. A declared
   * default lands on the parsed input, so these are the calls that would break if a
   * `.default()` disagreed with the flag it belongs to -- and the ones that show the
   * query Galaxy is actually asked for.
   */
  it("sends the declared defaults for a paged op without being told to", async () => {
    const asked: string[] = [];
    const run = await runCli(["list_pages", "--format", "json"], recordingContext(asked));
    const query = asking(asked, "/api/pages");
    expect(query).toContain("limit=100");
    expect(query).toContain("offset=0");
    expect(query).toContain("show_published=false");
    expect(query).toContain("show_shared=false");
    expect(run.exitCode).toBe(0);
  });

  it("still takes a flag over the declared default", async () => {
    const asked: string[] = [];
    const run = await runCli(
      ["list_pages", "--limit", "5", "--show-published", "--format", "json"],
      recordingContext(asked),
    );
    const query = asking(asked, "/api/pages");
    expect(query).toContain("limit=5");
    expect(query).toContain("show_published=true");
    expect(run.exitCode).toBe(0);
  });

  it("leaves list_workflows' published flag off the query unless it is asked for", async () => {
    // bioblend sends show_published only when it is true, so a defaulted false is an
    // absent parameter rather than an explicit one.
    const asked: string[] = [];
    const context = recordingContext(asked);
    await runCli(["list_workflows", "--format", "json"], context);
    expect(asking(asked, "/api/workflows")).not.toContain("show_published");
    asked.length = 0;
    await runCli(["list_workflows", "--published", "--format", "json"], context);
    expect(asking(asked, "/api/workflows")).toContain("show_published=true");
  });

  it("still converts a numeric flag for the ops that stopped coercing", async () => {
    // These three were the last schemas doing their own coercion; the conversion the
    // command line needs belongs in buildInput, and this is the proof it is there.
    const asked: string[] = [];
    const run = await runCli(["list_pages", "--offset", "7", "--format", "json"], recordingContext(asked));
    expect(asking(asked, "/api/pages")).toContain("offset=7");
    expect(run.exitCode).toBe(0);
  });

  it("refuses a numeric flag that is not a number, with a usage exit", async () => {
    const asked: string[] = [];
    const run = await runCli(["list_pages", "--limit", "abc", "--format", "json"], recordingContext(asked));
    expect(run.exitCode).toBe(64);
    expect(asked.some((url) => url.includes("/api/pages"))).toBe(false);
  });

  it("previews a dataset through the flags the op really declares", async () => {
    const asked: string[] = [];
    const context = () =>
      createGalaxyContext({
        baseUrl: "https://g.example",
        apiKey: "K",
        fetchImpl: (async (input: RequestInfo | URL) => {
          const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
          asked.push(url);
          const body = url.includes("get_content_as_text")
            ? { item_data: "l1\nl2\nl3\n", truncated: false, item_url: "/datasets/d1/display" }
            : { id: "d1", name: "reads.txt", state: "ok", file_ext: "txt" };
          return new Response(JSON.stringify(body), {
            status: 200,
            headers: { "content-type": "application/json" },
          });
        }) as typeof fetch,
      });

    const run = await runCli(["get_dataset_details", "d1", "--preview-lines", "2", "--format", "json"], context);
    expect(run.exitCode).toBe(0);
    expect(asking(asked, "get_content_as_text")).toContain("/api/datasets/d1/get_content_as_text");
    expect(run.stdout).toContain('"preview_lines": 2');
    expect(run.stdout).toContain('"truncated": true');

    // --no-include-preview is the other half of the boolean flag, and it must stop the fetch.
    asked.length = 0;
    const off = await runCli(
      ["get_dataset_details", "d1", "--no-include-preview", "--format", "json"],
      context,
    );
    expect(off.exitCode).toBe(0);
    expect(asked.some((url) => url.includes("get_content_as_text"))).toBe(false);
    expect(off.stdout).not.toContain('"preview"');
  });

  it("takes a list parameter as a repeatable flag rather than a positional", async () => {
    // A required array used to be offered as `<keywords>`, which could only ever arrive as one
    // string -- and `z.array(z.string())` refuses a string, so the command had no working form.
    const program = buildProgram({ makeContext: ctxFactory });
    const cmd = program.commands.find((c) => c.name() === "search_tools_by_keywords")!;
    expect(cmd.registeredArguments.map((a) => a.name())).toEqual([]);
    expect(cmd.options.map((o) => o.flags)).toContain("--keywords <value...>");
  });

  it("collects a repeated list flag into one array and runs the op", async () => {
    const asked: string[] = [];
    const panel = [
      { id: "bwa", name: "BWA", description: "Map low-divergent sequences" },
      { id: "cat1", name: "Concatenate", description: "datasets tail-to-head" },
    ];
    const run = await runCli(
      ["search_tools_by_keywords", "--keywords", "bwa", "--keywords", "concatenate", "--format", "json"],
      recordingContext(asked, panel),
    );
    expect(run.exitCode).toBe(0);
    expect(asking(asked, "/api/tools")).toContain("in_panel=true");
    expect(run.stdout).toContain('"success": true');
    expect(run.stdout).toContain('"bwa"');
    expect(run.stdout).toContain('"cat1"');
  });

  it("runs recommend_biocontainer off --packages, with quay.io stubbed", async () => {
    // The op reaches quay.io itself rather than through the injected context, so this
    // is the only place a CLI test has to stub the global fetch.
    const asked: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      asked.push(url);
      // A real Response, because the op streams the body to time out on inactivity.
      return new Response(
        JSON.stringify({ tags: { "1.17--h00cdaf9_0": { name: "1.17--h00cdaf9_0" } } }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    try {
      const run = await runCli(
        ["recommend_biocontainer", "--packages", "samtools=1.17", "--format", "json"],
        ctxFactory,
      );
      expect(run.exitCode).toBe(0);
      expect(asked[0]).toBe("https://quay.io/api/v1/repository/biocontainers/samtools");
      expect(run.stdout).toContain("quay.io/biocontainers/samtools:1.17--h00cdaf9_0");
      expect(run.stdout).toContain('"match_quality": "exact_version"');
      expect(run.stdout).toContain('"verified": true');
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("runs an op and renders json to stdout", async () => {
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    const program = buildProgram({ makeContext: ctxFactory });
    await program.parseAsync(["node", "galaxy-cli", "get_histories", "--format", "json"]);
    expect(out.mock.calls.flat().join("")).toContain('"success": true');
    expect(process.exitCode === 0 || process.exitCode === undefined).toBe(true);
    out.mockRestore();
  });
});

/**
 * The names this surface gives an op's inputs.
 *
 * The MCP server advertises the same parameters under the Python spelling (see
 * `galaxy-mcp/src/wire-names.ts`), and that rename stops there: a flag is the kebab-case of
 * the op's own key, a positional is the key itself, and neither moved when the other surface
 * was renamed. Derived here rather than imported from `flagName`, so a change to the
 * derivation is a failure rather than an agreement.
 */
describe("the names the CLI gives an op's inputs", () => {
  const commands = new Map(
    buildProgram({ makeContext: ctxFactory }).commands.map((c) => [c.name(), c]),
  );
  const kebab = (key: string) => key.replace(/[A-Z]/g, (m) => "-" + m.toLowerCase());

  it("names every flag after the op's own key, in kebab-case", () => {
    for (const op of allOperations) {
      const expected = Object.entries(op.input).flatMap(([key, schema]) => {
        const kind = classifyField(schema as ZodTypeAny);
        if (kind === "positional") return [];
        return kind === "boolean" ? [`--${kebab(key)}`, `--no-${kebab(key)}`] : [`--${kebab(key)}`];
      });
      expect(
        (commands.get(op.name)?.options ?? []).map((o) => o.long),
        op.name,
      ).toEqual(expected);
    }
  });

  // The optional fields kept as a positional from when they were required -- an agreement
  // with `keptPositionals`, written out so that adding one is a deliberate change here too.
  const KEPT: Record<string, string[]> = { get_job_details: ["datasetId"] };

  it("names every positional after the op's own key", () => {
    for (const op of allOperations) {
      const kept = KEPT[op.name] ?? [];
      const expected = Object.entries(op.input)
        .filter(([key, schema]) => classifyField(schema as ZodTypeAny) === "positional" || kept.includes(key))
        .map(([key]) => key);
      expect(
        (commands.get(op.name)?.registeredArguments ?? []).map((a) => a.name()),
        op.name,
      ).toEqual(expected);
    }
  });

  /**
   * And a name it does not know is refused here too -- commander's own behaviour, pinned
   * because it is the other half of the MCP surface closing its tool schemas: neither
   * surface may quietly ignore an argument somebody meant.
   *
   * `exitOverride` is the test's, not the CLI's: commander answers a usage error by writing
   * to stderr and exiting the process, which inside vitest would take the worker with it.
   * What is pinned is that it is a usage error at all, and that the op never runs.
   */
  it("refuses a flag the op does not have, without running anything", async () => {
    const asked: string[] = [];
    const program = buildProgram({ makeContext: recordingContext(asked) });
    for (const cmd of [program, ...program.commands]) {
      cmd.exitOverride();
      cmd.configureOutput({ writeErr: () => {}, writeOut: () => {} });
    }

    await expect(
      program.parseAsync(["node", "galaxy-cli", "get_histories", "--history-id", "h1"]),
    ).rejects.toMatchObject({ code: "commander.unknownOption" });
    expect(asked).toEqual([]);
  });

  /**
   * And the prose says the same thing the flags do. A description is written in the op's own
   * key -- `sectionId` -- because the ops are a TypeScript library; every surface respells it
   * on the way out, and a command line that says "pass sectionId" is naming something no
   * command line accepts. A positional keeps the key, spelled `<historyId>` the way the usage
   * line spells it, which is why the test looks for a BARE one.
   */
  it("names a parameter the way this surface takes it, in the help as well", () => {
    const camel = [
      ...new Set(allOperations.flatMap((op) => Object.keys(op.input).filter((k) => /[A-Z]/.test(k)))),
    ];
    const bare = (text: string) =>
      camel.filter((key) => new RegExp(`(?<![<\\w-])${key}\\b`).test(text));
    for (const op of allOperations) {
      const cmd = commands.get(op.name)!;
      expect(bare(cmd.description()), op.name).toEqual([]);
      for (const option of cmd.options) expect(bare(option.description), option.long).toEqual([]);
      for (const arg of cmd.registeredArguments) {
        expect(bare(arg.description), `${op.name} ${arg.name()}`).toEqual([]);
      }
    }
  });

  it("says the flag in the sentences that used to name the key", () => {
    // Commander wraps help to the terminal width, so the sentences are read unwrapped.
    const help = (name: string) => commands.get(name)!.helpInformation().replace(/\s+/g, " ");
    expect(help("get_tool_panel")).toContain("Pass --section-id to list one section");
    expect(help("get_tool_panel")).toContain("when --section-id is absent");
    expect(help("get_tool_panel")).toContain("to a --section-id call");
    expect(help("list_pages")).toContain("Pass --history-id to list only");
    expect(help("create_page")).toContain("With --history-id it is a notebook");
    expect(help("invoke_workflow")).toContain("ignored if --history-id is provided");
  });

  it("spells a few of them out, so the whole set cannot drift together", () => {
    const flags = (name: string) => (commands.get(name)?.options ?? []).map((o) => o.flags);
    expect(flags("get_tool_panel")).toContain("--section-id <value>");
    expect(flags("get_tool_details")).toContain("--io-details");
    expect(flags("download_dataset")).toContain("--file-path <value>");
    expect(flags("list_pages")).toContain("--show-published");
    expect(flags("get_page")).toContain("--include-rendered");
    expect(flags("list_page_revisions")).toContain("--sort-desc");
    expect((commands.get("get_history_details")?.registeredArguments ?? []).map((a) => a.name())).toEqual([
      "historyId",
    ]);
  });
});
