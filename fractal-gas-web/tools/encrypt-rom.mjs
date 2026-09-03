#!/usr/bin/env node
// Encrypt ROMs into the password-protected blobs the web app decrypts in
// the browser (web/main.js decryptRom). The password lives only in your
// head: the deployment ships ONLY the ciphertext, and nobody without the
// password can recover a ROM. This is real access control, not obfuscation.
//
// ONE password unlocks the whole vault: every bundled game (Mario, Sonic,
// Montezuma and the Atari picker) is encrypted with the same password and
// the same salt, so the browser derives the key once and then decrypts any
// ROM silently.
//
// Usage:
//   # Encrypt the whole vault (run from fractal-gas-web/): reads the
//   # plaintext ROMs next to the page (all gitignored) and writes
//   # web/roms-enc/**, which IS committed.
//   FG_ROM_PASSWORD='your-password' node tools/encrypt-rom.mjs --all
//
//   # Check that a password decrypts an existing blob (e.g. to reuse the
//   # current Sonic password before re-encrypting everything).
//   FG_ROM_PASSWORD='your-password' node tools/encrypt-rom.mjs --check web/roms-enc/sonic.rom.enc
//
//   # Encrypt a single file.
//   FG_ROM_PASSWORD='your-password' node tools/encrypt-rom.mjs web/sonic.rom web/roms-enc/sonic.rom.enc
//
// Blob layout (must match web/main.js): salt(16) | iv(12) | AES-256-GCM
// ciphertext-with-tag. Key = PBKDF2-SHA256(password, salt, 250000).
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { pbkdf2Sync, randomBytes, createCipheriv, createDecipheriv } from "node:crypto";

const ITERS = 250000;
const WEB = "web";
const ENC_DIR = join(WEB, "roms-enc");

const args = process.argv.slice(2);
const password = process.env.FG_ROM_PASSWORD;

function usage() {
  console.error(
    "Usage: FG_ROM_PASSWORD='...' node tools/encrypt-rom.mjs --all\n" +
    "       FG_ROM_PASSWORD='...' node tools/encrypt-rom.mjs --check <blob.enc>\n" +
    "       FG_ROM_PASSWORD='...' node tools/encrypt-rom.mjs <in.rom> <out.enc>");
  process.exit(1);
}
if (!password) usage();

const keyCache = new Map();
function deriveKey(salt) {
  const hex = salt.toString("hex");
  if (!keyCache.has(hex)) {
    keyCache.set(hex, pbkdf2Sync(password, salt, ITERS, 32, "sha256"));
  }
  return keyCache.get(hex);
}

function encrypt(plaintext, salt) {
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", deriveKey(salt), iv);
  const ct = Buffer.concat([cipher.update(plaintext), cipher.final()]);
  // WebCrypto AES-GCM expects the 16-byte auth tag appended to the ciphertext.
  return Buffer.concat([salt, iv, ct, cipher.getAuthTag()]);
}

function decrypt(blob) {
  const salt = blob.subarray(0, 16);
  const iv = blob.subarray(16, 28);
  const tag = blob.subarray(blob.length - 16);
  const ct = blob.subarray(28, blob.length - 16);
  const decipher = createDecipheriv("aes-256-gcm", deriveKey(salt), iv);
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(ct), decipher.final()]);  // throws on a wrong password
}

function encryptFile(inPath, outPath, salt) {
  const plaintext = readFileSync(inPath);
  const blob = encrypt(plaintext, salt);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, blob);
  console.log(`Wrote ${outPath} (${blob.length} bytes) from ${inPath} ` +
              `(${plaintext.length} bytes plaintext).`);
}

// The vault: plaintext location -> encrypted location (paths relative to
// fractal-gas-web/). Mirrors romSpec() in web/main.js.
function vaultEntries() {
  const entries = [
    { id: "mario", plain: join(WEB, "test-rom.nes"), enc: join(ENC_DIR, "mario.nes.enc") },
    { id: "sonic", plain: join(WEB, "sonic.rom"), enc: join(ENC_DIR, "sonic.rom.enc") },
  ];
  const atariDir = join(WEB, "roms", "atari");
  if (existsSync(atariDir)) {
    for (const file of readdirSync(atariDir).sort()) {
      if (!file.endsWith(".bin")) continue;
      entries.push({ id: `atari:${basename(file, ".bin")}`, plain: join(atariDir, file),
                     enc: join(ENC_DIR, "atari", `${file}.enc`) });
    }
  }
  return entries;
}

if (args[0] === "--all") {
  const salt = randomBytes(16);  // one salt for the whole vault
  const written = [];
  const missing = [];
  for (const e of vaultEntries()) {
    if (!existsSync(e.plain)) { missing.push(e); continue; }
    encryptFile(e.plain, e.enc, salt);
    written.push(e.id);
  }
  mkdirSync(ENC_DIR, { recursive: true });
  writeFileSync(join(ENC_DIR, "manifest.json"),
                JSON.stringify({ roms: written }, null, 2) + "\n");
  console.log(`Vault: ${written.length} ROMs encrypted into ${ENC_DIR}/ with one password.`);
  if (missing.length) {
    console.log("Skipped (plaintext not found): " +
                missing.map((e) => `${e.id} (${e.plain})`).join(", "));
  }
} else if (args[0] === "--check") {
  const blobPath = args[1];
  if (!blobPath) usage();
  try {
    const plain = decrypt(readFileSync(blobPath));
    console.log(`OK: the password decrypts ${blobPath} (${plain.length} bytes).`);
  } catch (err) {
    console.error(`FAIL: the password does not decrypt ${blobPath}.`);
    process.exit(2);
  }
} else {
  const [inPath, outPath] = args;
  if (!inPath || !outPath) usage();
  encryptFile(inPath, outPath, randomBytes(16));
}
