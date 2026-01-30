#!/usr/bin/env node
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { execSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const execFileAsync = promisify(execFile);

function resolveGeminiCliDeps() {
  const npmRoot = execSync('npm root -g', { encoding: 'utf8' }).trim();
  const base = path.join(npmRoot, '@google', 'gemini-cli', 'node_modules');
  const sdkBase = path.join(base, '@modelcontextprotocol', 'sdk', 'server');
  return {
    mcp: pathToFileURL(path.join(sdkBase, 'mcp.js')).href,
    stdio: pathToFileURL(path.join(sdkBase, 'stdio.js')).href,
    zod: pathToFileURL(path.join(base, 'zod', 'index.js')).href,
  };
}

function buildGeminiArgs(prompt, model) {
  const args = ['-p', prompt, '--output-format', 'json'];
  if (model) {
    args.push('--model', model);
  }
  return args;
}

async function runGemini(prompt, model) {
  const args = buildGeminiArgs(prompt, model);
  const { stdout } = await execFileAsync('gemini', args, {
    encoding: 'utf8',
    maxBuffer: 50 * 1024 * 1024,
  });
  const parsed = JSON.parse(stdout);
  if (parsed?.error) {
    const message = parsed.error.message || 'Gemini CLI returned an error.';
    throw new Error(message);
  }
  return parsed?.response ?? '';
}

const deps = resolveGeminiCliDeps();
const { McpServer } = await import(deps.mcp);
const { StdioServerTransport } = await import(deps.stdio);
const { z } = await import(deps.zod);

const server = new McpServer({
  name: 'gemini-cli',
  version: '1.0.0',
});

server.registerTool(
  'gemini_prompt',
  {
    description: 'Run a Gemini CLI prompt using OAuth auth from the local CLI.',
    inputSchema: z
      .object({
        prompt: z.string().min(1),
        model: z.string().optional(),
      })
      .shape,
  },
  async ({ prompt, model }) => {
    const text = await runGemini(prompt, model);
    return {
      content: [
        {
          type: 'text',
          text,
        },
      ],
    };
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
