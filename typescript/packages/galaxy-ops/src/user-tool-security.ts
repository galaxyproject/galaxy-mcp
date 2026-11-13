export interface UserToolSecurityDiagnostic {
  path: string;
  code: "dynamic_tool.unsafe_text_interpolation";
  message: string;
}

const TEMPLATE_BLOCK_RE = /\$\(([\s\S]*?)\)/g;
const DIRECT_INPUT_EXPR_RE = /^\s*inputs\.([A-Za-z_][A-Za-z0-9_]*)\s*$/;
const SCRIPT_SUFFIXES = [".bash", ".cjs", ".js", ".mjs", ".pl", ".py", ".r", ".rb", ".sh"];

function directInputRefs(value: unknown): Set<string> {
  const refs = new Set<string>();
  if (typeof value !== "string") return refs;
  for (const block of value.matchAll(TEMPLATE_BLOCK_RE)) {
    const expression = block[1];
    if (expression === undefined) continue;
    const match = DIRECT_INPUT_EXPR_RE.exec(expression);
    const name = match?.[1];
    if (name !== undefined) refs.add(name);
  }
  return refs;
}

export function userToolSecurityDiagnostics(
  tool: Record<string, unknown>,
): UserToolSecurityDiagnostic[] {
  const inputs = Array.isArray(tool.inputs) ? tool.inputs : [];
  const textInputs = new Set<string>();
  for (const input of inputs) {
    if (!input || typeof input !== "object" || Array.isArray(input)) continue;
    const record = input as Record<string, unknown>;
    if (record.type === "text" && typeof record.name === "string") textInputs.add(record.name);
  }
  const diagnostics: UserToolSecurityDiagnostic[] = [];
  const check = (value: unknown, path: string): void => {
    for (const name of directInputRefs(value)) {
      if (!textInputs.has(name)) continue;
      diagnostics.push({
        path,
        code: "dynamic_tool.unsafe_text_interpolation",
        message: `free-text input 'inputs.${name}' is substituted as executable text; serialize it into a config file with JSON.stringify and read it as data`,
      });
    }
  };
  check(tool.shell_command, "shell_command");
  const configfiles = Array.isArray(tool.configfiles) ? tool.configfiles : [];
  configfiles.forEach((configfile, index) => {
    if (!configfile || typeof configfile !== "object" || Array.isArray(configfile)) return;
    const record = configfile as Record<string, unknown>;
    const filename = record.filename;
    if (
      typeof filename === "string" &&
      SCRIPT_SUFFIXES.some((suffix) => filename.toLowerCase().endsWith(suffix))
    ) {
      check(record.content, `configfiles.${index}.content`);
    }
  });
  return diagnostics;
}
