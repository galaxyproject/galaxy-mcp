// Variables the code under test never reads itself but Node does: the proxy and TLS settings. With
// NODE_USE_ENV_PROXY=1 and a proxy set, fetch to the fake Galaxy on 127.0.0.1 goes to the proxy,
// and a CA or cert override changes what TLS trusts. A developer or CI runner behind a proxy saw
// tests fail, with nothing in the failures to point at the environment. Shared by
// global-setup.ts, which clears them before the workers start (Node reads NODE_USE_ENV_PROXY once,
// at startup, so clearing it later is too late), and safe-env.ts, which clears them again in each
// worker.
export const LEAKY_ENV = [
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "ALL_PROXY",
  "NO_PROXY",
  "http_proxy",
  "https_proxy",
  "all_proxy",
  "no_proxy",
  "NODE_USE_ENV_PROXY",
  "NODE_EXTRA_CA_CERTS",
  "SSL_CERT_FILE",
  "SSL_CERT_DIR",
] as const;
