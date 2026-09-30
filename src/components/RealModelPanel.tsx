import { useState } from "react";
import styles from "../pages/index.module.css";
import { RealGptSnapshot, runRealGpt } from "@/lib/realGpt";

export default function RealModelPanel() {
    const [status, setStatus] = useState("not loaded");
    const [snapshot, setSnapshot] = useState<RealGptSnapshot | null>(null);
    const [inputText, setInputText] = useState("The cat sat on the mat");

    const loadModel = async () => {
        setStatus("downloading GPT-2 weights...");
        try {
            setSnapshot(await runRealGpt(inputText));
            setStatus("loaded from Xenova/gpt2");
        } catch (error) {
            console.error(error);
            setStatus("load failed - check the browser console");
        }
    };

    return <section className={styles.realModel}><div><span className={styles.sectionLabel}>REAL CHECKPOINT</span><h2>Compare against GPT-2</h2><p>Load the GPT-2 checkpoint used by the browser runtime. This is separate from the tiny teaching matrices below: the tokenizer and logits here come from real GPT-2 weights.</p><input className={styles.realInput} value={inputText} onChange={(event) => setInputText(event.target.value)} aria-label="Text to tokenize with GPT-2" /></div><div className={styles.realAction}><button onClick={loadModel} disabled={status.startsWith("downloading") || !inputText.trim()}>{status.startsWith("downloading") ? "Loading..." : "Run GPT-2 on this text"}</button><span>{status}</span></div>{snapshot && <div className={styles.realSnapshot}><div className={styles.realTokens}><b>GPT-2 BPE encoding</b>{snapshot.tokens.map((token, index) => <span key={`${token}-${index}`}>{token || "[empty]"}<small>{snapshot.ids[index]}</small></span>)}</div><div className={styles.realPredictions}><b>Top logits for the next token</b>{snapshot.predictions.slice(0, 5).map((prediction) => <span key={prediction.id}><em>{prediction.token || "[empty]"}</em><small>id {prediction.id} / logit {prediction.logit.toFixed(3)}</small></span>)}</div></div>}</section>;
}
