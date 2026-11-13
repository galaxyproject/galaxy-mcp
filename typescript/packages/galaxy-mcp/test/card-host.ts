/**
 * A stand-in MCP Apps host for card tests: runs a card page's script against the few DOM pieces
 * it uses, and lets a test deliver host messages and click the card's buttons.
 */
import { runInNewContext } from "node:vm";

export const toolResult = (data: unknown, pagination: Record<string, unknown> | null = null) => ({
  content: [{ type: "text", text: JSON.stringify({ success: true, data, pagination }) }],
});

/** The few DOM pieces the page uses. Text is kept as plain strings among the children. */
export class Node {
  children: (Node | string)[] = [];
  attributes: Record<string, string> = {};
  listeners: Record<string, () => unknown> = {};
  className = "";
  type = "";
  disabled = false;
  constructor(readonly tag: string) {}
  append(...nodes: (Node | string)[]) {
    this.children.push(...nodes);
  }
  replaceChildren(...nodes: (Node | string)[]) {
    this.children = nodes;
  }
  addEventListener(type: string, listener: () => unknown) {
    this.listeners[type] = listener;
  }
  setAttribute(name: string, value: string) {
    this.attributes[name] = value;
  }
  get text(): string {
    return this.children.map((child) => (typeof child === "string" ? child : child.text)).join("");
  }
  all(match: (node: Node) => boolean): Node[] {
    const found: Node[] = [];
    for (const child of this.children) {
      if (typeof child === "string") continue;
      if (match(child)) found.push(child);
      found.push(...child.all(match));
    }
    return found;
  }
}

export function runCard(html: string) {
  const root = new Node("main");
  const sent: Record<string, any>[] = [];
  const parent = { postMessage: (message: Record<string, any>) => sent.push(message) };
  const listeners: ((event: { source: unknown; data: unknown }) => void)[] = [];
  const script = html.match(/<script>([\s\S]*)<\/script>/)![1]!;
  runInNewContext(script, {
    window: { parent, addEventListener: (type: string, f: (typeof listeners)[number]) => type === "message" && listeners.push(f) },
    document: {
      getElementById: () => root,
      createElement: (tag: string) => new Node(tag),
      documentElement: { style: {}, getBoundingClientRect: () => ({ height: 200 }) },
    },
    ResizeObserver: class {
      constructor(private readonly report: () => void) {}
      observe() {
        this.report();
      }
    },
    setTimeout,
    clearTimeout,
  });
  const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
  const card = {
    root,
    sent,
    deliver: async (data: Record<string, unknown>) => {
      for (const listener of listeners) listener({ source: parent, data: { jsonrpc: "2.0", ...data } });
      await settle();
    },
    initialize: (hostContext: Record<string, unknown> = {}) =>
      card.deliver({ id: sent[0]!.id, result: { hostCapabilities: { serverTools: {} }, hostContext } }),
    open: async (args: Record<string, unknown>, data: unknown, pagination: Record<string, unknown> | null) => {
      await card.deliver({ method: "ui/notifications/tool-input", params: { arguments: args } });
      await card.deliver({ method: "ui/notifications/tool-result", params: toolResult(data, pagination) });
    },
    button: (label: string) => root.all((n) => n.tag === "button" && n.text === label)[0],
    click: async (label: string) => {
      const button = card.button(label);
      if (!button) throw new Error(`no "${label}" button in: ${root.text}`);
      await button.listeners.click!();
      await settle();
    },
    rows: () => root.all((n) => n.tag === "li").map((li) => li.all((n) => n.className === "name")[0]!.children[0]),
    scope: () => root.all((n) => n.className === "scope")[0]!.text,
  };
  return card;
}
