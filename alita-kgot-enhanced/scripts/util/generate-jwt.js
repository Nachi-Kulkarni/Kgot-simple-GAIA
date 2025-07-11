#!/usr/bin/env node
/*
 * JWT Token Generator CLI
 *
 * Usage Examples:
 *   node scripts/util/generate-jwt.js --secret mySecret --payload '{"userId": 123}' --expiresIn 1h
 *   JWT_SECRET=mySecret node scripts/util/generate-jwt.js --payload '{"sub": "abc"}'
 *
 * Options:
 *   --secret      Secret key used to sign the token. If omitted, reads JWT_SECRET env var.
 *   --payload     JSON string representing the payload/claims. Defaults to `{}`.
 *   --expiresIn   Expiration time (e.g., 60s, 10m, 1h, 7d). Optional.
 *   --alg         Signing algorithm (default: HS256).
 *
 * The script prints the generated token to STDOUT. Errors are logged to STDERR with exit code 1.
 */

const jwt = require('jsonwebtoken');

// -------------------------
// Helper: basic arg parsing
// -------------------------
function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith('--')) {
      const key = arg.replace(/^--/, '');
      const next = argv[i + 1];
      if (next && !next.startsWith('--')) {
        args[key] = next;
        i++; // Skip next since it's a value
      } else {
        args[key] = true; // Boolean flag
      }
    }
  }
  return args;
}

(function main() {
  const argv = process.argv.slice(2);
  const args = parseArgs(argv);

  const secret = args.secret || process.env.JWT_SECRET;
  if (!secret) {
    console.error('Error: secret not provided. Supply via --secret or JWT_SECRET env variable.');
    process.exit(1);
  }

  let payload = {};
  if (args.payload) {
    try {
      payload = JSON.parse(args.payload);
    } catch (err) {
      console.error('Error: --payload must be valid JSON.');
      console.error(err.message);
      process.exit(1);
    }
  }

  const signOptions = {
    algorithm: args.alg || 'HS256',
  };
  if (args.expiresIn) {
    signOptions.expiresIn = args.expiresIn;
  }

  try {
    const token = jwt.sign(payload, secret, signOptions);
    console.log(token);
  } catch (err) {
    console.error('Failed to generate token:', err.message);
    process.exit(1);
  }
})(); 