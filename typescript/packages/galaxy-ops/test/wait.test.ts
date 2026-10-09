import { describe, it, expect } from "vitest";
import { waitForJob } from "../src/wait";
import { mockClient } from "./util/mock-client";
import { DEFAULT_POLL } from "../src/context";
import { JobFailedError } from "../src/errors";
import type { GalaxyContext } from "../src/context";

const fastPoll = { ...DEFAULT_POLL, intervalMs: 0, maxIntervalMs: 0, jitter: 0, timeoutMs: 5000 };
const ctxWith = (client: any): GalaxyContext => ({ client, poll: fastPoll });

function jobStates(states: string[]) {
  let i = 0;
  return mockClient({
    GET: () => {
      const state = states[Math.min(i, states.length - 1)]!;
      i++;
      return { data: { id: "j1", state }, response: { status: 200 } };
    },
  });
}

describe("waitForJob", () => {
  it("resolves with the terminal job on ok", async () => {
    const job = await waitForJob("j1", ctxWith(jobStates(["new", "running", "ok"])));
    expect(job.state).toBe("ok");
  });

  it("throws JobFailedError when the job reaches error", async () => {
    await expect(waitForJob("j1", ctxWith(jobStates(["running", "error"])))).rejects.toBeInstanceOf(
      JobFailedError,
    );
  });

  it("honors an aborted signal", async () => {
    const ac = new AbortController();
    ac.abort();
    const ctx: GalaxyContext = { client: jobStates(["new"]), poll: fastPoll, signal: ac.signal };
    await expect(waitForJob("j1", ctx)).rejects.toThrow(/abort/i);
  });

  it("times out if never terminal", async () => {
    const ctx = ctxWith(jobStates(["new"]));
    ctx.poll.timeoutMs = 5; // ms
    await expect(waitForJob("j1", ctx)).rejects.toThrow(/timed out/i);
  });
});
