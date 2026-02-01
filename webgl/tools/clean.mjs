#!/usr/bin/env node

import process from "node:process";
import console from "node:console";
import fs from 'node:fs/promises';
import path from "node:path";
import {fileURLToPath} from "url";

async function main() {
    const webgl_root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../');
    await Promise.all([
        fs.rm(path.resolve(webgl_root, 'dist'), {force: true, recursive: true}),
        fs.rm(path.resolve(webgl_root, 'src/ext'), {force: true, recursive: true})
    ])
}

if (process.argv[1] === import.meta.filename) {
    await main().catch(console.error);
}
