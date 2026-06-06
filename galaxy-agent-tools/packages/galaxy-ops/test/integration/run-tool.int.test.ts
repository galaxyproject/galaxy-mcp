import { describe, it, expect } from "vitest";
import { createGalaxyContext, getUser, runTool } from "../../src/index";

const URL = process.env.GALAXY_URL;
const KEY = process.env.GALAXY_API_KEY;
const HISTORY = process.env.GALAXY_TEST_HISTORY_ID; // a writable history on the target
const run = URL && KEY && HISTORY ? describe : describe.skip;

run("integration: runTool reaches terminal ok on a real Galaxy", () => {
  it("uploads-free tool runs to ok via the typed path", async () => {
    const ctx = createGalaxyContext({ baseUrl: URL!, apiKey: KEY! });
    const me = await getUser({}, ctx);
    expect(me.id).toBeTruthy();

    // Use a tool that needs no data input on the target (e.g. a text/parameter tool).
    // Override via env to match the deployment's tool ids.
    const toolId = process.env.GALAXY_TEST_TOOL_ID ?? "Show beginning1";
    const result = await runTool(
      { toolId, historyId: HISTORY!, inputs: JSON.parse(process.env.GALAXY_TEST_TOOL_INPUTS ?? "{}") },
      ctx,
    );
    expect(result.state).toBe("ok");
    expect(result.jobs.every((j) => (j as { state: string }).state === "ok")).toBe(true);
  }, 120_000);
});
