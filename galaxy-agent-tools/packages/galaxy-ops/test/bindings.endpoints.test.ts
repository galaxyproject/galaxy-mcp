import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const schema = readFileSync(
  fileURLToPath(new URL("../src/generated/schema.ts", import.meta.url)),
  "utf8",
);

describe("vendored bindings carry the tool-request surface", () => {
  for (const needle of [
    "/api/jobs",
    "/api/tool_requests/",
    "parameter_request_schema",
    "JobCreateResponse",
    "ToolRequestDetailedModel",
  ]) {
    it(`includes ${needle}`, () => {
      expect(schema.includes(needle)).toBe(true);
    });
  }
});
