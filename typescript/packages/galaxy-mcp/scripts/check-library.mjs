// Run after a build: imports this package by its own name, the way a program that installed it
// would, and checks the entry it gets. Node resolves the name through package.json's `exports`,
// so a missing file or a wrong path there fails here rather than for whoever installs the package.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
const functions = ["buildServer", "createHttpHandler", "startHttpServer", "httpConfig", "toolNames", "toolAnnotations"];
const types = ["ServerOptions", "HttpConfig", "HttpFlags", "HttpGuards", "HttpServerOptions"];
const problems = [];
try {
  const library = await import(pkg.name);
  problems.push(...functions.filter((name) => typeof library[name] !== "function").map((name) => `${name} is not exported as a function`));
} catch (error) {
  problems.push(`importing "${pkg.name}" failed: ${error.message}`);
}
try {
  const declarations = readFileSync(fileURLToPath(new URL(`../${pkg.exports["."].types}`, import.meta.url)), "utf8");
  problems.push(...[...functions, ...types].filter((name) => !new RegExp(`\\b${name}\\b`).test(declarations)).map((name) => `${name} is not declared in ${pkg.exports["."].types}`));
} catch (error) {
  problems.push(`reading the declarations failed: ${error.message}`);
}
if (problems.length) {
  console.error(`${pkg.name} does not import as a library:\n  ${problems.join("\n  ")}`);
  process.exit(1);
}
console.log(`${pkg.name} imports as a library: ${functions.length} functions, ${types.length} types`);
