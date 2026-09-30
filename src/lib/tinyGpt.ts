export type Vector = number[];
export type Matrix = number[][];

export const tokens = ["The", "cat", "sat", "on", "the", "mat"];
export const tokenIds = [83, 412, 921, 67, 83, 204];
export const vocabulary = ["floor", "mat", "sofa", "rug", "bed", "chair", "grass", "table"];

export const tokenEmbedding: Matrix = [
    [0.62, -0.14, 0.28, 0.41], [0.21, 0.71, -0.35, 0.18], [0.55, 0.09, 0.63, -0.22],
    [0.16, -0.40, 0.32, 0.76], [0.62, -0.14, 0.28, 0.41], [0.48, 0.34, -0.12, 0.57],
];
export const positionEmbedding: Matrix = [
    [0.10, 0, 0, 0], [0, 0.10, 0, 0], [0, 0, 0.10, 0], [0, 0, 0, 0.10], [0.08, 0.03, 0, 0], [0.03, 0.08, 0.02, 0],
];
export const Wq: Matrix = [[0.6, -0.2, 0.1, 0.3], [0.1, 0.7, -0.1, 0.2], [0.3, 0.1, 0.6, -0.2], [-0.2, 0.2, 0.3, 0.5]];
export const Wk: Matrix = [[0.5, 0.2, -0.1, 0.2], [-0.2, 0.6, 0.3, 0.1], [0.2, -0.1, 0.7, 0.2], [0.1, 0.3, 0.2, 0.6]];
export const Wv: Matrix = [[0.7, 0.1, 0.2, -0.1], [0.2, 0.6, -0.2, 0.1], [-0.1, 0.2, 0.5, 0.3], [0.2, -0.1, 0.3, 0.7]];
export const W1: Matrix = [[0.5, 0.2, -0.1, 0.3], [0.1, 0.6, 0.2, -0.2], [0.3, -0.2, 0.7, 0.1], [-0.1, 0.3, 0.2, 0.6]];
export const W2: Matrix = [[0.4, -0.1, 0.2, 0.3], [0.2, 0.5, 0.1, -0.2], [-0.2, 0.1, 0.6, 0.2], [0.3, 0.2, -0.1, 0.5]];
export const Wout: Matrix = [[0.7, -0.2, 0.4, 0.1, -0.1, 0.3, 0.2, -0.2], [0.1, 0.8, -0.2, 0.3, 0.2, -0.1, 0.4, 0.1], [-0.2, 0.1, 0.7, -0.1, 0.5, 0.2, -0.3, 0.4], [0.3, 0.2, 0.1, 0.6, 0.4, -0.2, 0.1, 0.5]];

export function multiply(left: Matrix, right: Matrix): Matrix {
    return left.map((row) => right[0].map((_, column) => row.reduce((sum, value, index) => sum + value * right[index][column], 0)));
}

export function add(left: Matrix, right: Matrix): Matrix {
    return left.map((row, rowIndex) => row.map((value, column) => value + right[rowIndex][column]));
}

export function relu(matrix: Matrix): Matrix {
    return matrix.map((row) => row.map((value) => Math.max(0, value)));
}

export function softmax(values: Vector): Vector {
    const max = Math.max(...values);
    const exponentials = values.map((value) => Math.exp(value - max));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map((value) => value / total);
}

export function rounded(value: number): string {
    return value.toFixed(3);
}

export type ForwardPass = {
    embedded: Matrix;
    queries: Matrix;
    keys: Matrix;
    values: Matrix;
    rawScores: Matrix;
    attention: Matrix;
    context: Matrix;
    residual: Matrix;
    ffHidden: Matrix;
    ffOutput: Matrix;
    blockOutput: Matrix;
    logits: Matrix;
    probabilities: Vector;
    layerCount: number;
};

export function runTinyGpt(temperature: number, layerCount = 1): ForwardPass {
    const embedded = add(tokenEmbedding, positionEmbedding);
    let current = embedded;
    let queries = embedded;
    let keys = embedded;
    let values = embedded;
    let rawScores: Matrix = [];
    let attention: Matrix = [];
    let context: Matrix = [];
    let residual = embedded;
    let ffHidden = embedded;
    let ffOutput = embedded;
    let blockOutput = embedded;
    for (let layer = 0; layer < layerCount; layer += 1) {
        queries = multiply(current, Wq);
        keys = multiply(current, Wk);
        values = multiply(current, Wv);
        rawScores = queries.map((query, rowIndex) => keys.map((key, columnIndex) => columnIndex > rowIndex ? Number.NEGATIVE_INFINITY : query.reduce((sum, value, index) => sum + value * key[index], 0) / 2));
        attention = rawScores.map((row) => softmax(row));
        context = attention.map((weights) => weights.map((_, dimension) => weights.reduce((sum, weight, index) => sum + weight * values[index][dimension], 0)));
        residual = add(current, context);
        ffHidden = relu(multiply(residual, W1));
        ffOutput = multiply(ffHidden, W2);
        blockOutput = add(residual, ffOutput);
        current = blockOutput;
    }
    const logits = multiply(blockOutput, Wout);
    const probabilities = softmax(logits[logits.length - 1].map((logit) => logit / temperature));
    return { embedded, queries, keys, values, rawScores, attention, context, residual, ffHidden, ffOutput, blockOutput, logits, probabilities, layerCount };
}
