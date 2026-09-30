export type RealGptSnapshot = {
    text: string;
    tokens: string[];
    ids: number[];
    predictions: { token: string; id: number; logit: number }[];
};

export async function runRealGpt(text: string): Promise<RealGptSnapshot> {
    const { AutoModelForCausalLM, AutoTokenizer } = await import("@xenova/transformers");
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2");
    const model = await AutoModelForCausalLM.from_pretrained("Xenova/gpt2", { quantized: true });
    const encoded = await tokenizer(text, { add_special_tokens: false });
    const ids = Array.from(encoded.input_ids.data as BigInt64Array, Number);
    const outputs = await model(encoded);
    const logits = outputs.logits;
    const vocabularySize = logits.dims[2];
    const lastTokenOffset = (logits.dims[1] - 1) * vocabularySize;
    const values = Array.from(logits.data as Float32Array).slice(lastTokenOffset, lastTokenOffset + vocabularySize);
    const predictions = values.map((logit, id) => ({ id, logit })).sort((left, right) => right.logit - left.logit).slice(0, 8).map(({ id, logit }) => ({
        id,
        logit,
        token: tokenizer.decode([id], { skip_special_tokens: false }),
    }));
    return {
        text,
        ids,
        tokens: ids.map((id) => tokenizer.decode([id], { skip_special_tokens: false })),
        predictions,
    };
}
