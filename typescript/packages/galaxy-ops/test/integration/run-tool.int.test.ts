import { describe, it, expect } from "vitest";
import { createGalaxyContext, getUser, runTool } from "../../src/index";

const URL = process.env.GALAXY_URL;
const KEY = process.env.GALAXY_API_KEY;
const HISTORY = process.env.GALAXY_TEST_HISTORY_ID; // a writable history on the target
const run = URL && KEY && HISTORY ? describe : describe.skip;

run("integration: runTool submits on a real Galaxy", () => {
  it("queues a tool run and comes back with the jobs Galaxy made", async () => {
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
    // A submission, not a wait: run_tool queues and answers, so what comes back is the
    // jobs in whatever state Galaxy started them in.
    expect(Array.isArray(result.jobs)).toBe(true);
    expect((result.jobs as unknown[]).length).toBeGreaterThan(0);
  }, 120_000);
});
