/**
 * Write `PARITY.md`, or print it.
 *
 * `--stdout` prints and writes nothing, which is what CI uses for the job summary:
 * it has to show the report the code produces, not whatever the checked-in file
 * happens to say.
 *
 * Run it as `pnpm parity:report` / `pnpm parity:report:print` rather than by path;
 * those scripts are what the failure messages name.
 */

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { REPORT_URL, currentReport } from "./report";

const text = await currentReport();

if (process.argv.slice(2).includes("--stdout")) {
  process.stdout.write(text);
} else {
  const path = fileURLToPath(REPORT_URL);
  writeFileSync(path, text);
  // stderr, so redirecting stdout still gets only the report.
  process.stderr.write(`wrote ${path}\n`);
}
