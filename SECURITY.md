# Security Policy

## Reporting a vulnerability

Please report security issues privately, not in a public issue or pull request.

The preferred route is GitHub's private vulnerability reporting: open the
[Security tab](https://github.com/galaxyproject/galaxy-mcp/security/advisories/new) of this
repository and choose "Report a vulnerability". If that doesn't work for you, email
galaxy-committers@lists.galaxyproject.org with `[SECURITY]` in the subject, as described in the
[Galaxy security policy](https://github.com/galaxyproject/galaxy/blob/dev/SECURITY.md).

Include what you found, where (file and line if you can), and how to reproduce it. We'll
acknowledge the report, work out a fix with you, and credit you in the advisory unless you'd
rather we didn't.

## Supported versions

Fixes go into the latest release of `galaxy-mcp` on PyPI and the matching container image.

## Deployment model

Galaxy MCP is designed first as a local server for the person running it. If you serve it over
HTTP, read [Serving over HTTP](mcp-server-galaxy-py/README.md#serving-over-http) for what the
server does and does not protect against.
