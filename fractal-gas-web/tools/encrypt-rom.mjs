#!/usr/bin/env node
// Encrypt a ROM into the password-protected blob the web app decrypts in the
// browser (web/main.js decryptRom). The password lives only in your head:
// the deployment ships ONLY this ciphertext, and nobody without the password
// can recover the ROM. This is real access control, not obfuscation.
//
// Usage:
//   FG_ROM_PASSWORD='your-password' node tools/encrypt-rom.mjs \
//       web/sonic.rom web/sonic.rom.enc
//
// Blob layout (must match web/main.js): salt(16) | iv(12) | AES-256-GCM
// ciphertext-with-tag. Key = PBKDF2-SHA256(password, salt, 250000).
import { readFileSync, writeFileSync } from "node:fs";
import { pbkdf2Sync, randomBytes, createCipheriv } from "node:crypto";

const ITERS = 250000;
const [inPath, outPath] = process.argv.slice(2);
const password = process.env.FG_ROM_PASSWORD;

if (!inPath || !outPath || !password) {
  console.error(
    "Usage: FG_ROM_PASSWORD='...' node tools/encrypt-rom.mjs <in.rom> <out.enc>");
  process.exit(1);
}

const plaintext = readFileSync(inPath);
const salt = randomBytes(16);
const iv = randomBytes(12);
const key = pbkdf2Sync(password, salt, ITERS, 32, "sha256");

const cipher = createCipheriv("aes-256-gcm", key, iv);
const ct = Buffer.concat([cipher.update(plaintext), cipher.final()]);
// WebCrypto AES-GCM expects the 16-byte auth tag appended to the ciphertext.
const blob = Buffer.concat([salt, iv, ct, cipher.getAuthTag()]);

writeFileSync(outPath, blob);
console.log(
  `Wrote ${outPath} (${blob.length} bytes) from ${inPath} ` +
  `(${plaintext.length} bytes plaintext).`);
