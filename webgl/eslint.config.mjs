import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import { globalIgnores } from "eslint/config";

export default tseslint.config(
    eslint.configs.recommended,
    tseslint.configs.recommended,
    globalIgnores([
        "node_modules/*",
        "webpack.*.js",
        "src/ext/*",
    ])
);
