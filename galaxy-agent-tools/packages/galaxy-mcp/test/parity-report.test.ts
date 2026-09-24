import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  compareSurfaces,
  type JsonSchema,
  type Normalization,
  type ToolContract,
} from "./parity/compare";
import {
  LINE_TERMINATOR,
  REGENERATE_COMMAND,
  cell,
  code,
  REPORT_URL,
  currentReport,
  renderReport,
  type ReportSurface,
} from "./parity/report";
import {
  loadRegistry,
  normalizationFrom,
  type AcceptedDivergence,
  type Registry,
} from "./parity/surfaces";

const REPORT_PATH = fileURLToPath(REPORT_URL);
const STALE =
  `PARITY.md is stale -- regenerate it with \`${REGENERATE_COMMAND}\` from ` +
  "`galaxy-agent-tools/`.";

/**
 * A report as its lines, the way both Git and a Markdown reader count them.
 *
 * Deliberately not the renderer's set. `cell` collapses every line terminator
 * Unicode has, because a cell must not carry one at all; but only these three end a
 * line in a GFM table, and only these three a checkout rewrites. Forgiving the rest
 * here would forgive a file where a newline had been swapped for U+2028 -- which a
 * reader does not break a row at, so the row after it disappears into the one before
 * while the comparison sees two tidy lines and calls the report current.
 */
const asLines = (text: string): string[] => text.split(/\r\n|[\n\r]/);

/** Where two reports first say something different, in words, or null when they do not. */
const firstDifference = (have: string, want: string): string | null => {
  const [found, wanted] = [asLines(have), asLines(want)];
  for (let line = 0; line < Math.max(found.length, wanted.length); line += 1) {
    if (found[line] === wanted[line]) continue;
    return (
      `line ${line + 1}: the file says ${JSON.stringify(found[line])}, ` +
      `the generator says ${JSON.stringify(wanted[line])}`
    );
  }
  return null;
};

/**
 * How many columns a row is, the way a markdown reader counts them: a pipe with a
 * backslash in front of it is content, and the pipes left over are the boundaries.
 */
const columnsIn = (line: string): number =>
  (line.replace(/\\\|/g, "").match(/\|/g) ?? []).length;

/** What a markdown reader takes back out again before it reads a cell. */
const unescaped = (text: string): string => text.replace(/\\\|/g, "|");

/**
 * A code span read the way a Markdown reader reads one: the fence is a run of
 * backticks at each end, and one space comes off each end when both are there. It
 * refuses anything that is not a code span, which is half of why it is called.
 */
const unfenced = (written: string): string => {
  const [opening] = /^`+/.exec(written) ?? [];
  const [closing] = /`+$/.exec(written) ?? [];
  expect(opening, `not a code span: ${written}`).toBeDefined();
  expect(closing?.length, `fences do not match: ${written}`).toBe(opening?.length);
  const inside = written.slice(opening?.length, written.length - (closing?.length ?? 0));
  const padded = inside.startsWith(" ") && inside.endsWith(" ") && inside.trim() !== "";
  return padded ? inside.slice(1, -1) : inside;
};

/**
 * Every table in a report, as its lines, first line being the headings. Rows are
 * broken where a reader breaks them, so a row the reader would swallow is one this
 * does not see either.
 */
const tablesIn = (text: string): string[][] => {
  const tables: string[][] = [];
  for (const line of asLines(text)) {
    if (!line.startsWith("|")) {
      if (tables.at(-1)?.length) tables.push([]);
      continue;
    }
    if (!tables.length) tables.push([]);
    tables.at(-1)?.push(line);
  }
  return tables.filter((rows) => rows.length);
};

/** The escapes a machine value is written with, read back. */
const decoded = (text: string): string =>
  text.replace(/\\(u[0-9a-f]{4}|.)/g, (_, escape: string) => {
    if (escape.startsWith("u")) return String.fromCharCode(Number.parseInt(escape.slice(1), 16));
    return { b: "\b", t: "\t", n: "\n", v: "\v", f: "\f", r: "\r" }[escape] ?? escape;
  });

/** Every row that is not the width its own table's headings set. */
const misshapenRows = (text: string): string[] =>
  tablesIn(text).flatMap(([headings, ...rest]) =>
    rest.filter((line) => columnsIn(line) !== columnsIn(headings as string)),
  );

/** The one table with a row per tool, whose column count the row tests are about. */
const toolTable = (text: string): string[] =>
  tablesIn(text).find((rows) => rows[0]?.startsWith("| Tool |")) ?? [];

/** The switches the check itself runs with, so a rendered shape is one CI compares. */
const RULES: Normalization = normalizationFrom(loadRegistry());

const registryOf = (...divergences: AcceptedDivergence[]): Registry => ({
  ...loadRegistry(),
  divergences,
});

const tool = (properties: Record<string, JsonSchema> = {}): ToolContract => ({
  inputSchema: { type: "object", properties },
  annotations: { readOnlyHint: true },
});

const columns = (...surfaces: [string, Record<string, ToolContract>][]): ReportSurface[] =>
  surfaces.map(([title, tools]) => ({ title, surface: new Map(Object.entries(tools)) }));

const render = (surfaces: ReportSurface[], registry: Registry): string => {
  const [python, typescript] = surfaces;
  const divergences =
    python && typescript
      ? compareSurfaces(python.surface, typescript.surface, RULES)
      : [];
  return renderReport({ surfaces, divergences, registry, rules: RULES });
};

describe("the checked-in parity report", () => {
  it("says what the generator says", async () => {
    const want = await currentReport();
    const have = readFileSync(REPORT_PATH, "utf8");
    expect(firstDifference(have, want), STALE).toBeNull();
  });

  it("is the same report whichever line ending the checkout gave it", async () => {
    // A Windows checkout, or `core.autocrlf=true` anywhere, hands the file back with
    // CRLF. That is not a stale report and must not read as one.
    const want = await currentReport();
    expect(firstDifference(want.replace(/\n/g, "\r\n"), want)).toBeNull();
    // And the comparison still has teeth: it is line endings it forgives, not content.
    expect(firstDifference(want.replace("# Surface parity", "# Surface disparity"), want)).toContain(
      "line 1",
    );
  });

  it("is rebuilt the same way twice", async () => {
    expect(await currentReport()).toBe(await currentReport());
  });

  it("has every row of every table under the headings it was written for", async () => {
    // The checked-in file, not what the generator would write: the file is what anyone
    // reads, and the two are only the same file while the check above says so.
    const committed = readFileSync(REPORT_PATH, "utf8");
    expect(tablesIn(committed).length, "no table was found in the report at all").toBeGreaterThan(
      1,
    );
    expect(misshapenRows(committed)).toEqual([]);
    // And it holds the rows the generator put there, so one lost to a reader shows up.
    const shape = (text: string) => tablesIn(text).map((rows) => rows.length);
    expect(shape(committed), STALE).toEqual(shape(await currentReport()));
  });

  it("would say so if a row in the file had been broken", () => {
    // The row after a newline swapped for U+2028 is not a row to a reader: it joins the
    // one before it, which is then wider than its headings. Done on the report's lines
    // rather than on its bytes, so a CRLF checkout is corrupted the same way and this
    // test is about the check rather than about how the file arrived.
    const committed = readFileSync(REPORT_PATH, "utf8");
    const breakARow = (text: string): string => {
      const lines = asLines(text).join("\n");
      const rowStart = lines.indexOf("\n| `connect`");
      expect(rowStart, "no row to break").toBeGreaterThan(-1);
      return `${lines.slice(0, rowStart)}\u2028${lines.slice(rowStart + 1)}`;
    };

    for (const arrival of [committed, asLines(committed).join("\r\n")]) {
      const broken = breakARow(arrival);
      expect(misshapenRows(broken).length).toBeGreaterThan(0);
      expect(firstDifference(broken, committed), "the stale check let it through").not.toBeNull();
    }
  });
});

describe("a cell of the report", () => {
  /** What the table forces: a line terminator cannot be carried, so it becomes a space. */
  const collapsed = (text: string): string => text.replace(LINE_TERMINATOR, " ");


  const TERMINATORS = [
    "\n",
    "\r",
    "\r\n",
    "\u000b",
    "\f",
    "\u0085",
    "\u2028",
    "\u2029",
  ];

  const NASTY = [
    "nothing here needs anything doing to it",
    "a | b",
    "run `printf '\\n'` first, and mind the \\d in the pattern",
    "a\\|b",
    "`a\\|b` beside `c|d`",
    "one \\\\| two",
    "two \\\\\\| three",
    "type=anyOf<object|string>",
    "it ends in a backslash \\",
    ...TERMINATORS.map((terminator) => `before${terminator}after`),
  ];

  it("has, as its source, the text it was given less what a table cannot carry", () => {
    for (const text of NASTY) {
      const written = cell(text);
      // The cell's SOURCE with the pipe escape undone, which is the text somebody
      // wrote, less the line break a row has no way to hold. What a reader then
      // displays is GFM's business -- it renders `\|` as `|`, and that is its call,
      // not something this has to reproduce.
      expect(unescaped(written), JSON.stringify(text)).toBe(collapsed(text));
      // And nothing is left in it that would end the cell or the row.
      expect(written.replace(/\\\|/g, ""), JSON.stringify(text)).not.toContain("|");
      expect(written.split(LINE_TERMINATOR).length, JSON.stringify(text)).toBe(1);
    }
  });

  const GENERATED = [
    "array&items<string>",
    "type=array&items<integer> required=false default=none",
    "a&b",
    "*x*",
    "_y_",
    "<!-- not a comment -->",
    "default=\"a`b\"",
    "default=\"a``b\"",
    "anyOf<object|string>",
    "read (tag), requires >=26.1",
    "`",
    "``",
  ];

  it("writes a value the comparison produced as itself, not as markup", () => {
    for (const value of GENERATED) {
      const written = cell(code(value));
      // A pipe still splits a cell inside a code span, so the escape goes on top and
      // comes off first; what is left is a code span holding exactly the value.
      expect(decoded(unfenced(unescaped(written))), JSON.stringify(value)).toBe(value);
      expect(written.replace(/\\\|/g, ""), JSON.stringify(value)).not.toContain("|");
    }
  });

  it("keeps two machine values apart, whatever is between them", () => {
    // A value the comparison called different from another has to LOOK different, and a
    // list of the characters that could hide one will always be missing the next. The
    // rule is printable ASCII or an escape, so this holds for anything at all.
    const between = [
      ...TERMINATORS,
      "\u0000",
      "\u007f",
      "\u00ad",
      "\u034f",
      "\u200b",
      "\u200e",
      "\u2060",
      "\ufeff",
      "\u3053\u3093\u306b\u3061\u306f",
      "\ud83d\ude00",
      "\ud800",
      " ",
    ];
    const cells = new Map<string, string>();
    for (const middle of between) {
      const value = `default="a${middle}b"`;
      const written = cell(code(value));
      const named = JSON.stringify(middle);
      // Printable ASCII, one line, and nothing that two different values could share.
      expect(written, named).toMatch(/^[\x20-\x7e]*$/);
      expect(written.split(LINE_TERMINATOR).length, named).toBe(1);
      expect(cells.get(written), `${named} renders as ${cells.get(written)}`).toBeUndefined();
      cells.set(written, named);
      // And it is still the value it was.
      expect(decoded(unfenced(unescaped(written))), named).toBe(value);
    }
    expect(cells.size).toBe(between.length);
    // An emoji is two escapes, because a cell is written in code units.
    expect(code("\ud83d\ude00")).toBe("`\\ud83d\\ude00`");
    // And the two characters `\n` are not the same value as a newline.
    expect(code("a\\nb")).not.toBe(code("a\nb"));
    expect(decoded(unfenced(code("a\\nb")))).toBe("a\\nb");
  });

  it("leaves a backslash alone, wherever it stands", () => {
    // Inside a code span a backslash is literal: doubling one rewrites the command a
    // reason is telling somebody to run.
    expect(cell("run `printf '\\n'`")).toBe("run `printf '\\n'`");
    expect(cell("a\\|b")).toBe("a\\\\|b");
    expect(cell("a\\\\|b")).toBe("a\\\\\\|b");
  });
});

describe("the report's tool table", () => {
  it("gives every tool a row and every surface a column", () => {
    const text = render(
      columns(["Python", { get_page: tool(), only_here: tool() }], ["TypeScript", { get_page: tool() }]),
      registryOf({
        tool: "only_here",
        param: null,
        kind: "missing-ts-tool",
        observed: "python=read (hint) params=[]",
        status: "pending-port",
        reason: "nobody has written the op yet",
      }),
    );

    expect(text).toContain("| Tool | Parameter | Python | TypeScript | Difference | Status | Why |");
    expect(text).toContain("| `get_page` |  | `read (hint)` | `read (hint)` |  |  |  |");
    expect(text).toContain(
      "| `only_here` |  | `read (hint)` | -- | `missing-ts-tool` | `pending-port` | " +
        "nobody has written the op yet |",
    );
  });

  it("takes a third surface as another column", () => {
    const text = render(
      columns(
        ["Python", { get_page: tool() }],
        ["TypeScript", { get_page: tool() }],
        ["Galaxy", {}],
      ),
      registryOf(),
    );

    expect(text).toContain(
      "| Tool | Parameter | Python | TypeScript | Galaxy | Difference | Status | Why |",
    );
    expect(text).toContain("| `get_page` |  | `read (hint)` | `read (hint)` | -- |  |  |  |");
  });

  it("says what each surface declares a diverging parameter to be", () => {
    const text = render(
      columns(
        ["Python", { get_page: tool({ limit: { type: "integer", default: 10 } }) }],
        ["TypeScript", { get_page: tool({ limit: { type: "string" } }) }],
      ),
      registryOf(
        {
          tool: "get_page",
          param: "limit",
          kind: "type-mismatch",
          observed: "python=integer typescript=string",
          status: "pending-decision",
          reason: "one of the two is wrong",
        },
        {
          tool: "get_page",
          param: "limit",
          kind: "default-mismatch",
          observed: "python=10 typescript=none",
          status: "unreviewed-gap",
          reason: "TS declares no default",
        },
      ),
    );

    // Both differences about one parameter share its row rather than splitting it,
    // in the order the comparison reports them.
    expect(text).toContain(
      "| `get_page` | `limit` | `type=integer required=false default=10` | " +
        "`type=string required=false default=none` | `default-mismatch`<br>`type-mismatch` | " +
        "`unreviewed-gap`<br>`pending-decision` | " +
        "TS declares no default<br>one of the two is wrong |",
    );
  });

  it("shows a parameter only one surface declares as absent from the other", () => {
    const text = render(
      columns(
        ["Python", { get_page: tool({ limit: { type: "integer" } }) }],
        ["TypeScript", { get_page: tool() }],
      ),
      registryOf({
        tool: "get_page",
        param: "limit",
        kind: "missing-ts-param",
        observed: "python=type=integer required=false default=none",
        status: "pending-port",
        reason: "not ported",
      }),
    );

    expect(text).toContain(
      "| `get_page` | `limit` | `type=integer required=false default=none` | -- | " +
        "`missing-ts-param` | `pending-port` | not ported |",
    );
  });

  it("marks a recorded assessment the surfaces have moved out from under", () => {
    // The entry still matches the tool, the parameter and the kind, so it is found; what
    // it recorded about the two sides is no longer true. Printing its reason as though it
    // were current tells whoever reads the summary the opposite of what happened.
    const withDefault = (value: unknown) => ({
      get_page: tool({ mode: { type: "string", default: value } }),
    });
    const text = render(
      columns(["Python", withDefault(false)], ["TypeScript", withDefault(true)]),
      registryOf({
        tool: "get_page",
        param: "mode",
        kind: "default-mismatch",
        observed: "python=true typescript=false",
        status: "pending-port",
        reason: "TS applies the same default in run() but does not declare it",
      }),
    );

    const row = toolTable(text).find((line) => line.includes("`mode`")) as string;
    expect(row).toContain("| `stale` |");
    expect(row, "the row does not say what the surfaces say now").toContain(
      "the surfaces now say `python=false typescript=true`",
    );
    expect(row, "the row does not say what was recorded").toContain(
      "the registry describes `python=true typescript=false` as `pending-port`",
    );
    expect(row).toContain("TS applies the same default in run() but does not declare it");
    // Both sides' current values are in their own columns either way.
    const [python, typescript] = row
      .split(" | ")
      .slice(2, 4)
      .map((written) => unfenced(unescaped(written)));
    expect(python).toBe("type=string required=false default=false");
    expect(typescript).toBe("type=string required=false default=true");
    // And it is counted as its own thing.
    expect(text).toContain("| `stale` | `1` |");
  });

  it("has no recorded assessment the surfaces have moved out from under", async () => {
    // A statement about today: every entry in the registry still describes the
    // difference it was written for, so no row of the real report reads `stale`.
    const report = await currentReport();
    const stale = toolTable(report).filter((row) => row.includes("| `stale` |"));
    expect(stale).toEqual([]);
    expect(report).not.toContain("| `stale` |");
  });

  it("marks a difference the registry has never heard of", () => {
    const text = render(
      columns(["Python", { get_page: tool() }], ["TypeScript", {}]),
      registryOf(),
    );

    expect(text).toContain("| `unregistered` | `1` |");
    expect(text).toMatch(
      /\| `get_page` \|  \| `read \(hint\)` \| -- \| `missing-ts-tool` \| `unregistered` \|/,
    );
    expect(text).toContain("the parity check fails until somebody reviews it");
  });

  it("counts the differences by status, and totals them", () => {
    const text = render(
      columns(["Python", { a_tool: tool(), b_tool: tool() }], ["TypeScript", {}]),
      registryOf(
        {
          tool: "a_tool",
          param: null,
          kind: "missing-ts-tool",
          observed: "python=read (hint) params=[]",
          status: "intentional",
          reason: "will not be ported",
        },
        {
          tool: "b_tool",
          param: null,
          kind: "missing-ts-tool",
          observed: "python=read (hint) params=[]",
          status: "pending-port",
          reason: "owed",
        },
      ),
    );

    expect(text).toContain("| `intentional` | `1` |");
    expect(text).toContain("| `pending-port` | `1` |");
    expect(text).toContain("| `pending-decision` | `0` |");
    expect(text).toContain("| **total** | `2` |");
    // The bucket for a difference nobody registered is left out when there are none.
    expect(text).not.toContain("| `unregistered` |");
  });

  it("does not let a reason break out of its cell", () => {
    const text = render(
      columns(["Python", { get_page: tool() }], ["TypeScript", {}]),
      registryOf({
        tool: "get_page",
        param: null,
        kind: "missing-ts-tool",
        observed: "python=read (hint) params=[]",
        status: "intentional",
        reason: "pipes | and\nnewlines belong to the prose, not to the table",
      }),
    );

    const [headings, , row] = toolTable(text);
    expect(row).toContain("pipes \\| and newlines belong to the prose, not to the table");
    expect(columnsIn(row as string), "the row grew a column").toBe(columnsIn(headings as string));
  });




  it("keeps an element type visible when that is the whole difference", () => {
    // `array&items<string>` read as Markdown loses its element type to an HTML tag,
    // and two cells that differ only there then read alike beside the word mismatch.
    const listOf = (element: string) => ({
      get_page: tool({ ids: { type: "array", items: { type: element } } }),
    });
    const text = render(
      columns(["Python", listOf("string")], ["TypeScript", listOf("integer")]),
      registryOf({
        tool: "get_page",
        param: "ids",
        kind: "type-mismatch",
        observed: "python=array&items<string> typescript=array&items<integer>",
        status: "unreviewed-gap",
        reason: "one of the two is wrong about what a list of ids holds",
      }),
    );

    const row = toolTable(text).find((line) => line.includes("`ids`"));
    // Take the fence and the pipe escape off each cell, so a cell that is not a code
    // span at all -- an element type that survives in the source and not on the page --
    // fails here.
    const [python, typescript] = (row as string)
      .split(" | ")
      .slice(2, 4)
      .map((written) => unfenced(unescaped(written)));
    expect(python).toBe("type=array&items<string> required=false default=none");
    expect(typescript).toBe("type=array&items<integer> required=false default=none");
    expect(python, "the two surfaces read alike on the row that says they differ").not.toBe(
      typescript,
    );
  });

  it("keeps two defaults apart when a separator is all that differs", () => {
    // `"a\u2028b"` against `"a b"` is a real default-mismatch, and both cells used to
    // read `default="a b"` -- the report agreeing on the row that says they differ.
    const withDefault = (value: string) => ({
      get_page: tool({ mode: { type: "string", default: value } }),
    });
    const text = render(
      columns(["Python", withDefault("a\u2028b")], ["TypeScript", withDefault("a b")]),
      registryOf({
        tool: "get_page",
        param: "mode",
        kind: "default-mismatch",
        observed: 'python="a\u2028b" typescript="a b"',
        status: "unreviewed-gap",
        reason: "one of the two has a separator in it",
      }),
    );

    const row = toolTable(text).find((line) => line.includes("`mode`"));
    const [python, typescript] = (row as string)
      .split(" | ")
      .slice(2, 4)
      .map((written) => unfenced(unescaped(written)));
    expect(python).toBe('type=string required=false default="a\\u2028b"');
    expect(typescript).toBe('type=string required=false default="a b"');
    expect(decoded(python as string)).toBe('type=string required=false default="a\u2028b"');
    expect(python, "the two cells read alike on the row that says they differ").not.toBe(
      typescript,
    );
  });

  it("does not let a union type break out of its cell", () => {
    // `anyOf<object|string>` is the comparison's own spelling, and the pipe in it is
    // not a column: a raw one puts every cell after it under the wrong heading.
    const text = render(
      columns(
        [
          "Python",
          { get_page: tool({ value: { anyOf: [{ type: "object" }, { type: "string" }] } }) },
        ],
        ["TypeScript", { get_page: tool({ value: { type: "object" } }) }],
      ),
      registryOf({
        tool: "get_page",
        param: "value",
        kind: "type-mismatch",
        observed: "python=anyOf<object|string> typescript=object",
        status: "unreviewed-gap",
        reason: "one of the two is wrong",
      }),
    );

    const rows = toolTable(text);
    const row = rows.find((line) => line.includes("`value`"));
    expect(row).toContain("`type=anyOf<object\\|string> required=false default=none`");
    expect(columnsIn(row as string), "the row grew a column").toBe(columnsIn(rows[0] as string));
  });

  it("sorts the tools, whatever order the surfaces list them in", () => {
    const [scrambled] = columns(["Python", { c_tool: tool(), a_tool: tool(), b_tool: tool() }]);
    const text = renderReport({
      surfaces: [scrambled as ReportSurface],
      divergences: [],
      registry: registryOf(),
      rules: RULES,
    });
    const tools = text
      .slice(text.indexOf("## Tools"))
      .split("\n")
      .filter((line) => line.startsWith("| `"))
      .map((line) => line.split(" | ")[0]);

    expect(tools).toEqual(["| `a_tool`", "| `b_tool`", "| `c_tool`"]);
  });
});
