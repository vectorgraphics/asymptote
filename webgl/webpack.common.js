const path = require('path');
const CopyPlugin = require('copy-webpack-plugin')
module.exports = {
    entry: './src/gl.ts',
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            },
            {
                test: /\.glsl$/,
                type: 'asset/source'
            },
        ],
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
    },
    output: {
        filename: 'gl.js',
        path: path.resolve(__dirname, 'dist'),
    }
};
