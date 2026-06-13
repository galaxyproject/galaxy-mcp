import { describe, it, expect } from "vitest";
import { createGalaxyContext, DEFAULT_POLL } from "../src/context";

describe("createGalaxyContext", () => {
  it("builds a client and applies poll defaults", () => {
    const ctx = createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K" });
    expect(ctx.client).toBeDefined();
    expect(ctx.poll).toEqual(DEFAULT_POLL);
  });
  it("allows poll overrides and passes through a signal", () => {
    const ac = new AbortController();
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      poll: { intervalMs: 50, timeoutMs: 1000 },
      signal: ac.signal,
    });
    expect(ctx.poll.intervalMs).toBe(50);
    expect(ctx.poll.timeoutMs).toBe(1000);
    expect(ctx.poll.backoff).toBe(DEFAULT_POLL.backoff); // unspecified -> default
    expect(ctx.signal).toBe(ac.signal);
  });
  it("stores baseUrl on the context for surfaces that need it", () => {
    const ctx = createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K" });
    expect(ctx.baseUrl).toBe("https://g.example");
  });
});
