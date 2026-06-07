import { describe, it, expect } from "vitest";
import { createGalaxyContext, getServerInfo, getHistories, createHistory, getToolDetails } from "../../src/index";

const URL = process.env.GALAXY_URL;
const KEY = process.env.GALAXY_API_KEY;
const run = URL && KEY ? describe : describe.skip;

run("integration: phase-1 ops on a real Galaxy", () => {
  const ctx = () => createGalaxyContext({ baseUrl: URL!, apiKey: KEY! });

  it("get_server_info returns a version", async () => {
    const info = await getServerInfo({}, ctx());
    expect(info.url).toBe(URL);
    expect(info.version).toBeTruthy();
  });

  it("get_histories returns an array", async () => {
    const hs = await getHistories({ limit: 5 }, ctx());
    expect(Array.isArray(hs)).toBe(true);
  });

  it("create_history actually creates one (form-encoded write proof)", async () => {
    const h = await createHistory({ historyName: `agent-tools phase1 ${Date.now()}` }, ctx());
    expect((h as { id?: string }).id).toBeTruthy();
  }, 30_000);

  it("get_tool_details reaches the legacy endpoint", async () => {
    const tool = await getToolDetails({ toolId: process.env.GALAXY_TEST_TOOL_ID ?? "cat1" }, ctx());
    expect(tool.id).toBeTruthy();
    expect(tool.name).toBeTruthy();
  });
});
