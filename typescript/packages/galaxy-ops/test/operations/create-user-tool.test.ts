import { describe, it, expect } from "vitest";
import { createUserToolOp, createUserTool } from "../../src/operations/create-user-tool";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyConnectionError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const VALID_REP = {
  class: "GalaxyUserTool",
  id: "my_filter",
  version: "0.1.0",
  name: "My Filter",
  shell_command: "echo hi",
  container: "python:3.12-slim",
};

describe("create_user_tool", () => {
  it("is a write op", () => {
    expect(createUserToolOp.readOnly).toBe(false);
  });

  it("throws without making a POST when a required field is missing", async () => {
    let postCalled = false;
    const client = mockClient({
      POST: () => {
        postCalled = true;
        return { data: {}, response: { status: 200 } };
      },
    });
    const { class: _cls, ...noClass } = VALID_REP;
    await expect(createUserTool({ representation: noClass }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyConnectionError,
    );
    expect(postCalled).toBe(false);
  });

  it("throws without making a POST when shell_command is missing (non-first required field)", async () => {
    let postCalled = false;
    const client = mockClient({
      POST: () => {
        postCalled = true;
        return { data: {}, response: { status: 200 } };
      },
    });
    const { shell_command: _sc, ...noShellCommand } = VALID_REP;
    await expect(
      createUserTool({ representation: noShellCommand }, ctxWith(client)),
    ).rejects.toThrow("shell_command");
    expect(postCalled).toBe(false);
  });

  it("throws when class is not GalaxyUserTool (no POST fired)", async () => {
    let postCalled = false;
    const client = mockClient({
      POST: () => {
        postCalled = true;
        return { data: {}, response: { status: 200 } };
      },
    });
    await expect(
      createUserTool({ representation: { ...VALID_REP, class: "WrongClass" } }, ctxWith(client)),
    ).rejects.toBeInstanceOf(GalaxyConnectionError);
    expect(postCalled).toBe(false);
  });

  it("throws when container is an object (no POST fired)", async () => {
    let postCalled = false;
    const client = mockClient({
      POST: () => {
        postCalled = true;
        return { data: {}, response: { status: 200 } };
      },
    });
    await expect(
      createUserTool({ representation: { ...VALID_REP, container: { image: "python:3.12-slim" } } }, ctxWith(client)),
    ).rejects.toBeInstanceOf(GalaxyConnectionError);
    expect(postCalled).toBe(false);
  });

  it("rejects direct free-text interpolation without making a POST", async () => {
    let postCalled = false;
    const client = mockClient({
      POST: () => {
        postCalled = true;
        return { data: {}, response: { status: 200 } };
      },
    });
    const representation = {
      ...VALID_REP,
      shell_command: "printf '%s' '$(inputs.label)'",
      inputs: [{ name: "label", type: "text" }],
    };
    await expect(createUserTool({ representation }, ctxWith(client))).rejects.toThrow(
      "dynamic_tool.unsafe_text_interpolation",
    );
    expect(postCalled).toBe(false);
  });

  it("accepts free text serialized into a data config", async () => {
    const representation = {
      ...VALID_REP,
      shell_command: "python script.py params.json",
      inputs: [{ name: "label", type: "text" }],
      configfiles: [
        {
          name: "params",
          filename: "params.json",
          content: '{"label": $(JSON.stringify(inputs.label))}',
          eval_engine: "ecmascript",
        },
      ],
    };
    const client = mockClient({
      POST: () => ({ data: { uuid: "abc" }, response: { status: 200 } }),
    });
    await expect(createUserTool({ representation }, ctxWith(client))).resolves.toMatchObject({
      uuid: "abc",
    });
  });

  it("POSTs valid representation with correct body shape and returns created tool", async () => {
    const client = mockClient({
      POST: (path, init) => {
        expect(path).toBe("/api/unprivileged_tools");
        expect(init.body.src).toBe("representation");
        expect(init.body.representation).toEqual(VALID_REP);
        return { data: { id: "1", uuid: "abc-123", tool_id: "my_filter/0.1.0", active: true }, response: { status: 200 } };
      },
    });
    const out = await createUserTool({ representation: VALID_REP }, ctxWith(client));
    expect(out.uuid).toBe("abc-123");
    expect(out.active).toBe(true);
  });

  it("names the representation it was given, not the record Galaxy answered with", () => {
    const created = { id: "1", uuid: "abc-123", tool_id: "my_filter/0.1.0", active: true };
    expect(createUserToolOp.project!(created, { representation: VALID_REP })).toEqual({
      message: `Created user-defined tool '${String(VALID_REP["name"])}'`,
    });
  });

  it("falls back to the representation's id, and then to unknown", () => {
    const created = { uuid: "abc-123" };
    const { name: _dropped, ...noName } = VALID_REP as Record<string, unknown>;
    expect(createUserToolOp.project!(created, { representation: noName }).message).toBe(
      `Created user-defined tool '${String(noName["id"])}'`,
    );
    expect(createUserToolOp.project!(created, { representation: {} }).message).toBe(
      "Created user-defined tool 'unknown'",
    );
    // Present and null is not absent, so it is announced rather than defaulted.
    expect(createUserToolOp.project!(created, { representation: { name: null } }).message).toBe(
      "Created user-defined tool 'None'",
    );
  });
});
