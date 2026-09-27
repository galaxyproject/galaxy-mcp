import { beforeAll, describe, expect, it, vi } from "vitest";
import { build, type BuildFailure, type BuildOptions, type Metafile } from "esbuild";
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { pathToFileURL } from "node:url";
// Only the browser entry, and nothing else: the registry is one module-level array, so a
// file that also imported the full entry would be asserting against a registry both filled.
import { allOperations as browserOperations } from "../src/index.browser";

/**
 * The browser entry is a promise about what a bundle can do, so it is checked by making a
 * bundle rather than by reading the source. Nothing here pattern-matches an import. A
 * `node:` specifier can be a side-effect import, a dynamic one, single-quoted, or a module
 * or two away, and each of those walks past a reader that greps while breaking the bundle
 * all the same. esbuild resolves the graph a consumer's bundler will resolve and refuses a
 * node builtin outright when the platform is a browser, so "can this run in a browser" is
 * asked of the bundler and the answer is whatever it says.
 */
const PKG_DIR = join(__dirname, "..");
const OPS_DIR = join(PKG_DIR, "src", "operations");
const REGISTRY = join(OPS_DIR, "registry.ts");

/**
 * What a consumer's bundler resolves for itself: this package's declared runtime
 * dependencies, read from package.json so the list cannot go stale. Everything else is
 * bundled -- our own source, and anything a module reaches for that we never declared --
 * which is what leaves a node builtin nowhere to hide. `node:crypto` is not a declared
 * dependency, so it is not external, so a browser build fails on it.
 */
const EXTERNAL = Object.keys(
  (JSON.parse(readFileSync(join(PKG_DIR, "package.json"), "utf8")) as { dependencies: Record<string, string> })
    .dependencies,
).flatMap((dep) => [dep, `${dep}/*`]);

const BROWSER_BUILD: BuildOptions = {
  bundle: true,
  platform: "browser",
  format: "esm",
  write: false,
  metafile: true,
  logLevel: "silent",
  external: EXTERNAL,
  // Makes the metafile's input keys package-relative, so they can be compared by name.
  absWorkingDir: PKG_DIR,
};

type Bundled = { ok: true; text: string; metafile: Metafile } | { ok: false; errors: string };

/** Build something for a browser, reporting a refusal instead of throwing it. */
async function bundleForBrowser(what: BuildOptions): Promise<Bundled> {
  try {
    const result = await build({ ...BROWSER_BUILD, ...what });
    return { ok: true, text: result.outputFiles![0]!.text, metafile: result.metafile! };
  } catch (error) {
    return { ok: false, errors: ((error as BuildFailure).errors ?? []).map((e) => e.text).join("; ") };
  }
}

/**
 * The modules a module imports directly, as esbuild resolved them.
 *
 * Built for node, because this asks what the file imports and not whether a browser can run
 * it -- a browser build of a list that has just gone wrong fails, and then there is no graph
 * to compare against and no way to name which member went wrong.
 */
async function directImports(entry: string): Promise<string[]> {
  const result = await build({ ...BROWSER_BUILD, platform: "node", entryPoints: [entry] });
  const key = Object.keys(result.metafile!.inputs).find((k) => k.endsWith(`/${basename(entry)}`))!;
  return result
    .metafile!.inputs[key]!.imports.filter((i) => i.path.startsWith("src/operations/"))
    .map((i) => basename(i.path, ".ts"))
    .sort();
}

/** A synthetic module built as if it sat in src/operations, without ever being written there. */
function asOpModule(contents: string): BuildOptions {
  return { stdin: { contents, resolveDir: OPS_DIR, sourcefile: "synthetic-op.ts", loader: "ts" } };
}

/**
 * The ops a module registers when it is the only thing imported into a fresh registry.
 *
 * This is how an op module is told from a helper: not by finding a `register(` in the text,
 * where indentation or a rename hides it, but by importing the module and seeing whether
 * the registry grew. A module that registers nothing is a helper and has no business in
 * all-browser.ts; a module that registers something is an op and belongs there unless a
 * browser cannot run it.
 */
async function registrations(moduleUrl: string): Promise<string[]> {
  vi.resetModules();
  const registry = await import("../src/operations/registry.ts");
  const before = registry.allOperations.length;
  await import(/* @vite-ignore */ moduleUrl);
  return registry.allOperations.slice(before).map((op) => op.name);
}

/** Every module in src/operations that is a candidate for the browser list. */
const candidates = readdirSync(OPS_DIR)
  .filter((f) => f.endsWith(".ts") && f !== "all.ts" && f !== "all-browser.ts")
  .map((f) => basename(f, ".ts"))
  .sort();

/** What all-browser.ts imports -- the one thing still read from source, and the thing under test. */
let listed: string[];
/** Candidates whose import registers at least one op. */
let opModules: string[];
/** Op modules that bundle for a browser, and the ones that do not. */
let browserSafe: string[];
let nodeOnly: string[];
const registeredBy = new Map<string, string[]>();

beforeAll(async () => {
  listed = await directImports(join(OPS_DIR, "all-browser.ts"));

  const safe: string[] = [];
  const unsafe: string[] = [];
  for (const name of candidates) {
    const file = join(OPS_DIR, `${name}.ts`);
    registeredBy.set(name, await registrations(pathToFileURL(file).href));
    const built = await bundleForBrowser({ entryPoints: [file] });
    (built.ok ? safe : unsafe).push(name);
  }
  opModules = candidates.filter((name) => registeredBy.get(name)!.length > 0);
  browserSafe = opModules.filter((name) => safe.includes(name));
  nodeOnly = opModules.filter((name) => unsafe.includes(name));
}, 60_000);

describe("the browser entry", () => {
  it("bundles for a browser with no node builtin left in it", async () => {
    const built = await bundleForBrowser({ entryPoints: [join(PKG_DIR, "src", "index.browser.ts")] });
    expect(built.ok ? "" : built.errors).toBe("");
    expect(built.ok ? built.text : "").not.toMatch(/node:/);
  });

  it("leaves out exactly the op modules a browser cannot run", () => {
    // Derived on both sides: the set that bundles against the set all-browser.ts imports.
    // An op that grows a node dependency anywhere in its graph drops out of browserSafe; a
    // new browser-safe op nobody listed turns up in browserSafe and not in listed.
    expect(listed.length).toBeGreaterThan(0);
    expect(nodeOnly.length).toBeGreaterThan(0);
    expect(browserSafe).toEqual(listed);
    expect(nodeOnly).toEqual(opModules.filter((name) => !listed.includes(name)));
  });

  it("registers exactly the ops its listed modules register", () => {
    const expected = [...new Set(listed.flatMap((name) => registeredBy.get(name)!))].sort();
    expect(browserOperations.map((op) => op.name).sort()).toEqual(expected);
  });

  it("registers no op that needs a filesystem or node:crypto", () => {
    const names = browserOperations.map((op) => op.name);
    expect(names.length).toBeGreaterThan(0);
    expect(names).not.toContain("upload_file");
    expect(names).not.toContain("download_dataset");
    expect(names).not.toContain("recommend_biocontainer");
  });

  it("catches a node builtin however it is written", async () => {
    const control = await bundleForBrowser(asOpModule(`export const op = 1;\n`));
    expect(control.ok ? "" : control.errors).toBe("");

    // The forms a scan for `from "node:` misses. Each is the control plus one node builtin,
    // so the refusal can only be the builtin.
    const sideEffect = await bundleForBrowser(asOpModule(`import "node:crypto";\nexport const op = 1;\n`));
    expect(sideEffect.ok ? "" : sideEffect.errors).toContain("node:crypto");

    const dynamic = await bundleForBrowser(
      asOpModule(`export const op = async () => await import("node:crypto");\n`),
    );
    expect(dynamic.ok ? "" : dynamic.errors).toContain("node:crypto");

    const singleQuoted = await bundleForBrowser(asOpModule(`import 'node:crypto';\nexport const op = 1;\n`));
    expect(singleQuoted.ok ? "" : singleQuoted.errors).toContain("node:crypto");
  });

  it("catches a node builtin a module reaches through a helper", async () => {
    // What a per-file scan cannot see at all, and the reason recommend_biocontainer is on the
    // Node side: the builtin is a module away. Both files live outside the package.
    const dir = mkdtempSync(join(tmpdir(), "galaxy-ops-browser-entry-"));
    try {
      writeFileSync(join(dir, "helper.ts"), `import "node:crypto";\nexport const hash = 1;\n`, "utf8");
      const entry = join(dir, "synthetic-op.ts");
      writeFileSync(entry, `import "./helper";\nexport const op = 1;\n`, "utf8");
      const built = await bundleForBrowser({ entryPoints: [entry] });
      expect(built.ok ? "" : built.errors).toContain("node:crypto");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("counts an op whose register() call is indented", async () => {
    // A scan anchored on `^register(` misses this one; importing the module does not care
    // where on the line the call sits. Written outside the package so nothing in src/ moves
    // and the real registry keeps its own contents.
    const dir = mkdtempSync(join(tmpdir(), "galaxy-ops-browser-entry-"));
    try {
      const file = join(dir, "synthetic-op.ts");
      writeFileSync(
        file,
        [
          `import { register } from ${JSON.stringify(pathToFileURL(REGISTRY).href)};`,
          `  register({`,
          `    name: "synthetic_indented_op",`,
          `    domain: "histories",`,
          `    summary: "an op whose register() call is not at the start of a line",`,
          `    input: {},`,
          `    run: async () => ({}),`,
          `  });`,
          ``,
        ].join("\n"),
        "utf8",
      );
      expect(await registrations(pathToFileURL(file).href)).toEqual(["synthetic_indented_op"]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
