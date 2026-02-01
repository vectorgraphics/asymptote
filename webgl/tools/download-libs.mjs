#!/usr/bin/env node

import process from 'node:process';
import {mkdir} from 'node:fs/promises';
import console from 'node:console';
import path from "node:path";
import fs from 'node:fs/promises'
import {fileURLToPath} from 'url';

const tinyexr_wasm_url = "https://cdn.jsdelivr.net/gh/jamievlin/tinyexr-js-build@52a0d072c508574feb6e0f7ba7e80848f7a09c3a/tinyexr.wasm";
const tinyexr_js_url = "https://cdn.jsdelivr.net/gh/jamievlin/tinyexr-js-build@52a0d072c508574feb6e0f7ba7e80848f7a09c3a/tinyexr.js";

/**
 *
 * @param {string} url
 * @param {string} path
 * @returns {Promise<void>}
 */
async function downloadFile(url, path) {
    const res = await fetch(url); // eslint-disable-line no-undef
    if (res.ok) {
        const blob = await res.blob();
        await fs.writeFile(path, await blob.bytes());
    } else {
        throw new Error(`Downloading from ${url} failed`)
    }
}

async function main() {
    const webgl_root = path.dirname(fileURLToPath(import.meta.url));
    const tinyexr_path = path.resolve(webgl_root, '../src/ext/tinyexr/');
    await mkdir(tinyexr_path, {recursive: true});
    await downloadFile(tinyexr_js_url, path.resolve(tinyexr_path, 'tinyexr.js'));
    await downloadFile(tinyexr_wasm_url, path.resolve(tinyexr_path, 'tinyexr.wasm'));
}

if (process.argv[1] === import.meta.filename) {
    await main().catch(console.error);
}
