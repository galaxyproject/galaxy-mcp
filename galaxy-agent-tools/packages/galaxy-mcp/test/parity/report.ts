/**
 * The parity report: `PARITY.md` at the repository root, built from the surfaces
 * the check compares and the registry that records what it found.
 *
 * The registry is the enforcement half, and nobody opens a JSON file to find out
 * how the two surfaces are doing. This renders the same facts as something a diff
 * can show: the file is checked in, a test holds it to what the generator says,
 * and a pull request that moves parity moves the file in the same commit.
 *
 * It is built around a LIST of surfaces rather than two named ones, because a
 * third is coming -- Galaxy's own built-in MCP server -- and it should arrive as
 * another column and more divergences, not as a second report. What is still
 * pairwise is the divergence kinds: `missing-ts-param` names two sides, so a third
 * surface will have to say which pair each of its divergences is about.
 */

import {
  compareSurfaces,
  divergenceKey,
  mutability,
  normalizeParams,
  requirement,
  showContract,
  showMutability,
  type Divergence,
  type NormalParam,
  type Normalization,
  type Surface,
} from "./compare";
import {
  DIVERGENCE_STATUSES,
  RATCHETED_STATUS,
  builtinRegistry,
  builtinSide,
  builtinSurface,
  loadBuiltinSnapshot,
  loadFixtureCounts,
  loadManifest,
  loadRegistry,
  normalizationFrom,
  pythonSurface,
  ratchetCeiling,
  typescriptSurface,
  type AcceptedDivergence,
  type BuiltinProvenance,
  type Registry,
} from "./surfaces";

/** Where the report lives, and the one command that rewrites it. */
export const REPORT_URL = new URL("../../../../../PARITY.md", import.meta.url);
export const REGENERATE_COMMAND = "pnpm parity:report";

/** What a cell says when the surface does not have the thing the row is about. */
const ABSENT = "--";

/** The status of a divergence the registry has never heard of. */
const UNREGISTERED = "unregistered";

/**
 * The status of a divergence the registry describes as something it no longer is. A
 * status is somebody's reading of what the two surfaces said, and it is worth no more
 * than that reading is current.
 */
const STALE = "stale";

/** One column of the report: a surface and what to call it. */
export interface ReportSurface {
  title: string;
  surface: Surface;
}

/**
 * The second pair: this repository's Python surface against the one Galaxy serves.
 *
 * Optional, because the snapshot it reads needs a Galaxy checkout and a report without it
 * is still a true report of the two surfaces here. Kept beside the first pair rather than
 * merged into it because the two say different things: the pair above is a promise about
 * what this repository releases, and this one is a list of questions for Galaxy.
 */
export interface ReportBuiltin {
  /** Which Galaxy the third column describes, printed so a reader can date it. */
  galaxy: BuiltinProvenance;
  divergences: readonly Divergence[];
  accepted: readonly AcceptedDivergence[];
  /** How many unreviewed gaps this pair's own ratchet allows. */
  ratchet: number;
}

export interface ReportInput {
  surfaces: readonly ReportSurface[];
  divergences: readonly Divergence[];
  registry: Registry;
  /** The switches the comparison ran with, so the report shows the shapes it compared. */
  rules: Normalization;
  builtin?: ReportBuiltin;
  /**
   * How many golden cases each tool has: how much of its answer is compared against the
   * other server at all. Absent in a hand-built report, where every count reads zero.
   */
  caseCounts?: ReadonlyMap<string, number>;
}

/**
 * Every line terminator Unicode has. A row ends at one of these, so a cell that
 * carries one carries the rest of its text into a row of its own. Exported because
 * anything reading a rendered report back has to break rows on the same set, and a
 * second copy of it would eventually disagree with this one.
 *
 * Written as a pattern over an ASCII string rather than as a literal: several of
 * these characters end a line in source too, and a literal carrying them does not
 * parse.
 */
export const LINE_TERMINATOR = new RegExp(
  "\\r\\n|[\\n\\v\\f\\r\\u0085\\u2028\\u2029]",
  "g",
);

const ESCAPES: Record<string, string> = {
  "\\": "\\\\",
  "\b": "\\b",
  "\t": "\\t",
  "\n": "\\n",
  "\v": "\\v",
  "\f": "\\f",
  "\r": "\\r",
};

/**
 * A machine value as printable ASCII, every other character written the way a JSON
 * string writes it.
 *
 * Not the space that prose gets. Two values the comparison called different have to
 * LOOK different, and the only way to promise that is to stop deciding which
 * characters hide: a default of `"a b"` beside one with a line separator in it is a
 * real divergence, and so is one with a combining grapheme joiner in it, and a list
 * of the ones somebody thought of will always be missing the next. Everything
 * outside 0x20-0x7e goes out as an escape instead, which makes the rendering
 * one-to-one by construction. A value in another script pays for that by arriving as
 * escapes, and stays exact and readable back.
 *
 * Code units, not code points, so an emoji is two escapes and a lone surrogate is
 * still writable. The backslash escapes itself in the same pass, or a value holding
 * the two characters `\n` would read back as one holding a newline.
 */
const visible = (value: string): string =>
  value.replace(
    /[^\x20-\x7e]|\\/g,
    (unit) =>
      ESCAPES[unit] ?? `\\u${unit.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );

/**
 * Text that cannot break out of the table cell it is written into, changed no more
 * than the table grammar forces. Every cell goes through this, not just the prose:
 * a union type reads `anyOf<object|string>`, and a raw pipe there is an extra
 * column and every cell after it in the wrong one.
 *
 * Two characters and nothing else. A row splits on a pipe, and a pipe written with
 * a backslash in front of it is not a split -- however many backslashes stand
 * before that one. Everything else is left exactly as the surface or the reason
 * wrote it: a cell may hold a code span, and inside one a backslash is literal, so
 * touching it would change what the text says.
 *
 * The line terminator becoming a space is the PROSE path. A machine value has
 * already had its own written out as escapes by `code`, because two values the
 * comparison called different have to look different; a reason is somebody's words,
 * where a space is what a line break means.
 */
export const cell = (text: string): string =>
  text.replace(LINE_TERMINATOR, " ").replace(/\|/g, "\\|");

/**
 * A value the report built rather than a person wrote, as a code span.
 *
 * Everything in these tables that came out of the comparison -- a type token, a
 * declared default, a divergence kind, a status, what a surface says about changing
 * things -- is text about code, and read as Markdown it is not itself. `array&items<string>`
 * loses its element type to an HTML tag; a default of `*` or `_x_` turns into emphasis;
 * `&lt;` is a character reference. A code span makes all of it literal, and the one
 * character a code span cannot hold on its own is the pipe, which `cell` escapes on
 * top of this because a GFM table splits on one wherever it sits.
 *
 * Anything in the value a reader could not see is written out as an escape first,
 * so the cell is one line and two values that differ only there still differ here.
 *
 * The fence is a backtick run longer than any run inside the value, which is how a
 * code span carries backticks at all. A space at each end is added when the value
 * begins or ends with a backtick, or with a space at both ends, because a reader
 * takes one space off each end when it finds them -- so the padding is what survives
 * being read rather than the value's own edges.
 */
export const code = (value: string): string => {
  if (!value.trim()) return value;
  const shown = visible(value);
  const longest = Math.max(0, ...[...shown.matchAll(/`+/g)].map((run) => run[0].length));
  const fence = "`".repeat(longest + 1);
  const padded = /^`|`$/.test(shown) || (shown.startsWith(" ") && shown.endsWith(" "));
  const pad = padded ? " " : "";
  return `${fence}${pad}${shown}${pad}${fence}`;
};

const stacked = (values: readonly string[]): string => values.join("<br>");

const row = (cells: readonly string[]): string => `| ${cells.map(cell).join(" | ")} |`;

const table = (headings: readonly string[], rows: readonly string[][]): string[] => [
  row(headings),
  row(headings.map(() => "---")),
  ...rows.map(row),
];

const byText = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** Where a divergence sits, without the kind: one row can carry several kinds. */
const at = (d: { tool: string; param: string | null }): string => `${d.tool} :: ${d.param ?? ""}`;

/** Everything a surface says about a whole tool that the comparison compares. */
function toolCell(column: ReportSurface, tool: string): string {
  const contract = column.surface.get(tool);
  if (!contract) return ABSENT;
  const where = `${tool} (${column.title})`;
  const needs = requirement(contract, where);
  return code(`${showMutability(mutability(contract, where))}${needs ? `, requires ${needs}` : ""}`);
}

export function renderReport(input: ReportInput): string {
  const { surfaces, divergences, registry, rules, builtin, caseCounts } = input;
  const tools = [...new Set(surfaces.flatMap((s) => [...s.surface.keys()]))].sort(byText);
  /**
   * The pairs this report covers, each with the rows it found and the entries somebody
   * wrote about them. A built-in row says so in its `Difference` cell: the kinds name two
   * sides and a cell holding rows from both pairs would otherwise read as one comparison.
   */
  const pairs = [
    { prefix: "", divergences, accepted: registry.divergences },
    ...(builtin
      ? [{ prefix: "vs built-in: ", divergences: builtin.divergences, accepted: builtin.accepted }]
      : []),
  ].map((pair) => ({
    ...pair,
    entries: new Map(pair.accepted.map((d) => [divergenceKey(d), d])),
  }));

  /**
   * What the registry says about a divergence, and what to say when what it says is
   * not about this divergence any more.
   *
   * An entry is matched on tool, parameter and kind, and it also records what each
   * side said at the time. When that has moved, the status and the reason are somebody
   * reading a difference that no longer exists: the surfaces have changed under the
   * entry, the parity check is already failing over it, and a summary that printed the
   * old explanation as though it were current would be telling whoever reads it the
   * opposite of what happened. Such a row says `stale` and shows the reading it came
   * from beside the values the surfaces give now.
   */
  const verdict = (
    d: Divergence,
    entries: Map<string, AcceptedDivergence>,
  ): { status: string; reason: string } => {
    const entry = entries.get(divergenceKey(d));
    if (!entry) {
      return {
        status: UNREGISTERED,
        // The only reasons nobody wrote: what the surfaces said is the comparison's own
        // words, so it goes in as code, and the sentence around it is the prose part.
        reason:
          `${code(d.observed)} -- nothing in the registry describes this, and the parity ` +
          "check fails until somebody reviews it",
      };
    }
    if (entry.observed !== d.observed) {
      return {
        status: STALE,
        reason:
          `the surfaces now say ${code(d.observed)}, and the registry describes ` +
          `${code(entry.observed)} as ${code(entry.status)}: ${entry.reason}`,
      };
    }
    return { status: entry.status, reason: entry.reason };
  };

  // Normalized once per tool per surface, so every parameter row of a tool reads the
  // list the comparison read rather than one parsed again for each row. Keyed by the
  // column itself, so there is no index that could be out of range.
  const parsed = new Map<ReportSurface, Map<string, Map<string, NormalParam>>>(
    surfaces.map((column) => [column, new Map()]),
  );
  const parametersOf = (column: ReportSurface, tool: string) => {
    const cache = parsed.get(column);
    const already = cache?.get(tool);
    if (already) return already;
    const contract = column.surface.get(tool);
    const params = contract
      ? normalizeParams(contract.inputSchema, rules, `${tool} (${column.title})`)
      : new Map<string, NormalParam>();
    cache?.set(tool, params);
    return params;
  };

  /** Every row either pair found, in pair order, by the place it is about. */
  const found = new Map<string, { kind: string; status: string; reason: string }[]>();
  for (const pair of pairs) {
    for (const d of pair.divergences) {
      const { status, reason } = verdict(d, pair.entries);
      found.set(at(d), [
        ...(found.get(at(d)) ?? []),
        { kind: `${pair.prefix}${d.kind}`, status, reason },
      ]);
    }
  }
  /**
   * The last three cells of a row: what differs, what somebody called it, and why.
   * The first two are the comparison's own words and go in as code; the reason is
   * prose a person wrote for a reader, so it stays prose.
   */
  const verdicts = (where: string): string[] => {
    const here = found.get(where) ?? [];
    return [
      stacked(here.map((d) => code(d.kind))),
      stacked(here.map((d) => code(d.status))),
      stacked(here.map((d) => d.reason)),
    ];
  };

  /**
   * Which parameters of a tool get a row: the ones EITHER pair found a difference
   * about, not just the first pair's.
   *
   * Read off every pair for the same reason the rows are: a row is how a difference
   * reaches a reader, and a difference the summary counts and the table leaves out is
   * worse than one nobody had found -- the count says it was looked at. The test that
   * holds the parameter rows to the comparator's own list is what keeps this honest.
   */
  const diverging = new Map<string, Set<string>>();
  for (const pair of pairs) {
    for (const d of pair.divergences) {
      if (d.param === null) continue;
      const already = diverging.get(d.tool) ?? new Set<string>();
      diverging.set(d.tool, already.add(d.param));
    }
  }

  // A row per tool, then a row per parameter the surfaces disagree about. A whole-tool
  // divergence has no parameter of its own, so it is reported on the tool's own row.
  const rows: string[][] = [];
  for (const tool of tools) {
    rows.push([
      code(tool),
      "",
      code(String(caseCounts?.get(tool) ?? 0)),
      ...surfaces.map((column) => toolCell(column, tool)),
      ...verdicts(`${tool} :: `),
    ]);
    const parameters = [...(diverging.get(tool) ?? [])].sort(byText);
    for (const param of parameters) {
      rows.push([
        code(tool),
        code(param),
        // A count is a fact about the tool, and a parameter's row is not another tool.
        "",
        ...surfaces.map((column) => {
          const declared = parametersOf(column, tool).get(param);
          return declared ? code(showContract(declared)) : ABSENT;
        }),
        ...verdicts(`${tool} :: ${param}`),
      ]);
    }
  }

  /** One pair's count of a status, and the statuses worth a row for that pair. */
  const counter = (pair: (typeof pairs)[number]) => {
    const held = (status: string): number =>
      pair.divergences.filter((d) => verdict(d, pair.entries).status === status).length;
    // The two statuses nobody chose are only worth a row when there is one to count.
    const statuses = [...DIVERGENCE_STATUSES, UNREGISTERED, STALE].filter(
      (status) => ![UNREGISTERED, STALE].includes(status) || held(status) > 0,
    );
    return { held, statuses, total: pair.divergences.length };
  };
  const statusTable = (pair: (typeof pairs)[number]): string[] => {
    const { held, statuses, total } = counter(pair);
    return table(
      ["Status", "Differences"],
      [
        ...statuses.map((status) => [code(status), code(String(held(status)))]),
        ["**total**", code(String(total))],
      ],
    );
  };
  const [first, second] = pairs;

  return [
    "# Surface parity",
    "",
    `Generated -- do not edit by hand. Regenerate with \`${REGENERATE_COMMAND}\` from ` +
      "`galaxy-agent-tools/`, which is also what CI checks this file against.",
    "",
    "Three surfaces expose the same Galaxy operations, and this is every way they disagree. " +
      "The Python column is the checked-in surface manifest " +
      "(`mcp-server-galaxy-py/tests/testdata/mcp-surface.json`); the TypeScript column is what " +
      "a client is really advertised by `@galaxyproject/galaxy-mcp`" +
      (builtin
        ? "; the Built-in column is the MCP server Galaxy itself serves " +
          "(`lib/galaxy/webapps/galaxy/api/mcp.py`, behind `enable_mcp_server`), read from a " +
          "snapshot of that module"
        : "") +
      ". Compared: which tools " +
      "exist, what parameters they take, their types, requiredness and declared defaults, " +
      "whether a tool takes parameters it does not declare, whether it says it changes " +
      "anything, and what it says it needs from the server. " +
      "Not compared: result shapes, wording, value constraints, what is inside an object, and " +
      "everything else -- so a difference can be real and have no row here.",
    "",
    "Every difference carries the status and the reason recorded in " +
      "`galaxy-agent-tools/packages/galaxy-mcp/test/fixtures/accepted-divergences.json`, which " +
      "is also where the statuses themselves are explained. A status says how well a difference " +
      "is understood, not that it is acceptable.",
    "",
    ...(builtin
      ? [
          `The third column describes Galaxy \`${builtin.galaxy.describe}\` ` +
            `(\`${builtin.galaxy.commit.slice(0, 11)}\` on \`${builtin.galaxy.branch}\`, ` +
            `committed ${builtin.galaxy.commitDate.slice(0, 10)}, captured ` +
            `${builtin.galaxy.capturedOn}). That column is a snapshot taken by hand, because ` +
            "reading it needs a Galaxy checkout and CI has none: it is as current as the commit " +
            "it names and no more. Refresh it with `builtin_surface.py`, which says how.",
          "",
          "Two differences with that server are recorded once here rather than as a row per " +
            "tool, because they are true of all of them: every built-in tool takes the API key " +
            "as an argument, where both surfaces here take the credential once per session; and " +
            "every built-in tool is registered without tags or a read-only hint, so a client is " +
            "told nothing about which of them change state and MCP's own default is to assume " +
            "they all might. That is why the Built-in column reads `write (mcp default)` " +
            "throughout.",
          "",
        ]
      : []),
    "## Differences by status",
    "",
    ...(second ? ["### Python and TypeScript", ""] : []),
    ...statusTable(first!),
    "",
    `\`${RATCHETED_STATUS}\` is the status nobody has ruled on yet. The check holds the registry ` +
      `to the ${ratchetCeiling(registry)} it declares, so the count cannot drift from the number; ` +
      "raising that number is an edit somebody has to make in the diff, and it is meant to come " +
      "down, never up.",
    "",
    ...(second && builtin
      ? [
          "### Python and Galaxy's built-in server",
          "",
          ...statusTable(second),
          "",
          "These are counted apart and ratcheted apart -- at " +
            `${builtin.ratchet} -- because they are not this repository's to close on its own: ` +
            "each one is a rename, an addition or a removal somebody has to agree with " +
            "galaxyproject/galaxy. Nothing here has been ruled on yet.",
          "",
        ]
      : []),
    ...(surfaces.length > 1
      ? [
          "## What each surface is missing",
          "",
          "Every tool any surface has, against the surfaces that do not have it. A name here " +
            "is either a tool somebody has not written yet or the same tool under another " +
            "name; the rows further down say which, one tool at a time.",
          "",
          ...surfaces.flatMap((column) => {
            const absent = tools.filter((tool) => !column.surface.has(tool));
            return [
              `**Missing from ${column.title}** (${absent.length}): ` +
                (absent.length ? absent.map((tool) => code(tool)).join(", ") : "nothing"),
              "",
            ];
          }),
        ]
      : []),
    "## Tools",
    "",
    `A row per tool, then a row per parameter the surfaces disagree about. \`${ABSENT}\` means ` +
      "that surface does not have it. A tool's own row says what it advertises about changing " +
      "things and about the Galaxy it needs; a parameter's row says what each surface declares " +
      "it to be, in the terms the comparison compares." +
      (second ? " A difference found against the built-in server says so in its kind." : ""),
    "",
    "`Cases` is how many golden result envelopes that tool has under " +
      "`mcp-server-galaxy-py/tests/testdata/envelopes` -- calls the Python server answered, " +
      "replayed through both surfaces here and compared key for key. It is not part of the " +
      "comparison above, which is about what a tool declares; it is how much of what a tool " +
      "ANSWERS anybody checks. A `0` on a tool that is not in the registry's `fixtures.excluded` " +
      "list fails the parity check.",
    "",
    ...table(
      ["Tool", "Parameter", "Cases", ...surfaces.map((s) => s.title), "Difference", "Status", "Why"],
      rows,
    ),
    "",
  ].join("\n");
}

/**
 * The three surfaces and both pairs as they stand.
 *
 * Separate from the rendering so a test can hold the rendered table to the same
 * comparison the file was written from, rather than to a second wiring of the loaders
 * that could drift away from this one.
 */
export async function currentInput(): Promise<ReportInput> {
  const registry = loadRegistry();
  const rules = normalizationFrom(registry);
  const python = pythonSurface(loadManifest());
  const typescript = await typescriptSurface();
  const snapshot = loadBuiltinSnapshot();
  const section = builtinRegistry(registry);
  const builtin = builtinSurface(snapshot, section);
  return {
    surfaces: [
      { title: "Python", surface: python },
      { title: "TypeScript", surface: typescript },
      { title: "Built-in", surface: builtin },
    ],
    divergences: compareSurfaces(python, typescript, rules),
    registry,
    rules,
    caseCounts: loadFixtureCounts(),
    builtin: {
      galaxy: snapshot.galaxy,
      divergences: compareSurfaces(python, builtin, rules, builtinSide(section)),
      accepted: section.divergences,
      ratchet: section.ratchet.unreviewedGaps,
    },
  };
}

/** The report for the surfaces as they stand, which is what the checked-in file has to match. */
export async function currentReport(): Promise<string> {
  return renderReport(await currentInput());
}
