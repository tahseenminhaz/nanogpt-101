import { useState } from "react";
import dynamic from "next/dynamic";
import styles from "./index.module.css";
import {
    Matrix,
    Wk,
    Wout,
    Wq,
    Wv,
    ForwardPass,
    runTinyGpt,
    rounded,
    positionEmbedding,
    tokenEmbedding,
    tokenIds,
    tokens,
    vocabulary,
} from "@/lib/tinyGpt";

const RealModelPanel = dynamic(() => import("@/components/RealModelPanel"), { ssr: false });

type Stage = "input" | "attention" | "block" | "prediction";

const stages: { id: Stage; number: string; name: string; description: string }[] = [
    { id: "input", number: "01", name: "Embed", description: "Token IDs are looked up and added to position vectors." },
    { id: "attention", number: "02", name: "Attend", description: "Q, K, and V turn earlier tokens into a weighted context." },
    { id: "block", number: "03", name: "Transform", description: "Residual connections and the MLP refine each token row." },
    { id: "prediction", number: "04", name: "Predict", description: "The final row becomes logits and a probability distribution." },
];

function MatrixTable({ matrix, rowLabels, colLabels, focusRow, masked }: { matrix: Matrix; rowLabels?: string[]; colLabels?: string[]; focusRow?: number; masked?: boolean }) {
    return (
        <div className={styles.matrixWrap}>
            {colLabels && <div className={styles.matrixHeader}><span />{colLabels.map((label) => <span key={label}>{label}</span>)}</div>}
            {matrix.map((row, rowIndex) => <div className={`${styles.matrixRow} ${rowIndex === focusRow ? styles.matrixRowFocus : ""}`} key={`row-${rowIndex}`}>
                {rowLabels && <span className={styles.matrixLabel}>{rowLabels[rowIndex]}</span>}
                {row.map((value, columnIndex) => <span className={`${styles.matrixCell} ${masked && !Number.isFinite(value) ? styles.maskedCell : ""}`} key={`cell-${rowIndex}-${columnIndex}`}>{Number.isFinite(value) ? rounded(value) : "-"}</span>)}
            </div>)}
        </div>
    );
}

function Vector({ values, label }: { values: number[]; label: string }) {
    return <div className={styles.vectorReadout}><span>{label}</span>{values.map((value, index) => <b key={`${label}-${index}`}>{rounded(value)}</b>)}</div>;
}

function StageCard({ id, active, title, subtitle, children, onClick }: { id: Stage; active: boolean; title: string; subtitle: string; children: React.ReactNode; onClick: () => void }) {
    return <article className={`${styles.stageCard} ${active ? styles.stageCardActive : ""}`} onClick={onClick}>
        <div className={styles.cardHeading}><span className={styles.cardNumber}>{stages.find((stage) => stage.id === id)?.number}</span><div><span className={styles.cardEyebrow}>{subtitle}</span><h2>{title}</h2></div><span className={styles.cardState}>{active ? "INSPECTING" : "VIEW"}</span></div>
        {children}
    </article>;
}

function Formula({ children }: { children: React.ReactNode }) {
    return <div className={styles.formula}>{children}</div>;
}

function ParameterPanel({ title, matrix, note }: { title: string; matrix: Matrix; note: string }) {
    return <div className={styles.parameterPanel}><div className={styles.parameterHeading}><strong>{title}</strong><span>learned weights</span></div><MatrixTable matrix={matrix} /><p>{note}</p></div>;
}

function TraceStep({ number, title, formula, shape, matrix, rowLabels, colLabels, masked }: { number: string; title: string; formula: string; shape: string; matrix: Matrix; rowLabels?: string[]; colLabels?: string[]; masked?: boolean }) {
    return <article className={styles.traceStep}><div className={styles.traceStepHead}><span>{number}</span><div><strong>{title}</strong><code>{formula}</code></div><b>{shape}</b></div><MatrixTable matrix={matrix} rowLabels={rowLabels} colLabels={colLabels} masked={masked} /></article>;
}

export default function Home() {
    const [activeStage, setActiveStage] = useState<Stage>("attention");
    const [focusRow, setFocusRow] = useState(5);
    const [selectedLayer, setSelectedLayer] = useState(1);
    const [temperature, setTemperature] = useState(0.8);
    const pass: ForwardPass = runTinyGpt(temperature, 4);
    const layerState = pass.layers[selectedLayer - 1];
    const previousState = selectedLayer === 1 ? pass.embedded : pass.layers[selectedLayer - 2].blockOutput;
    const topPredictions = pass.probabilities.map((probability, index) => ({ token: vocabulary[index], probability })).sort((left, right) => right.probability - left.probability).slice(0, 5);
    const rowLabels = tokens.map((token) => token.trim());
    const focusVector = pass.blockOutput[focusRow];
    const selectedStage = stages.find((stage) => stage.id === activeStage) ?? stages[1];

    return <main className={styles.shell}>
        <header className={styles.header}><div className={styles.brand}><span className={styles.brandMark}>n/</span><span>nanoGPT <em>101</em></span></div><div className={styles.headerMeta}><span className={styles.statusDot} /> Fixed toy weights / computed pass <span className={styles.divider} /><a href="https://github.com/karpathy/nanoGPT" target="_blank" rel="noreferrer">reference repo -&gt;</a></div></header>

        <section className={styles.hero}><div><p className={styles.kicker}>Numbers in motion</p><h1>Trace one prediction<br /><i>from token to thought.</i></h1></div><p className={styles.heroCopy}>This is a deliberately tiny GPT. Every matrix is visible, every number is computed in your browser, and every stage can be inspected without hiding the algebra.</p></section>

        <section className={styles.path}><div className={styles.sectionLabel}><span>TEACHING FORWARD PASS</span><span>6 tokens / 4 features / 1 head</span></div><div className={styles.pathRail}>{stages.map((stage) => <button key={stage.id} className={`${styles.pathStep} ${activeStage === stage.id ? styles.pathStepActive : ""}`} onClick={() => setActiveStage(stage.id)}><span>{stage.number}</span><strong>{stage.name}</strong></button>)}</div><p className={styles.stageHint}><b>{selectedStage.name}:</b> {selectedStage.description}</p></section>

        <RealModelPanel />

        <section className={styles.inputBar}><div><span className={styles.inputLabel}>TEACHING MODEL INPUT</span><strong>What comes after &quot;The cat sat on the mat&quot;?</strong><small className={styles.inputNote}>These compact IDs belong to the four-feature teaching model. The real GPT-2 BPE IDs are shown above.</small></div><div className={styles.tokenButtons}>{tokens.map((token, index) => <button className={focusRow === index ? styles.tokenActive : ""} key={`${token}-${index}`} onClick={() => setFocusRow(index)}>{token}<small>id {tokenIds[index]}</small></button>)}</div></section>

        <section className={styles.flow}>
            <div className={styles.flowHeader}><div><span className={styles.sectionLabel}>MODEL FLOW</span><h2>Follow the numbers</h2><p className={styles.layerNote}>Teaching stack layer {selectedLayer} of 4. Each selected layer repeats the same Transformer block formula with a new input state.</p></div><div className={styles.focusControl}><span>layer</span><select value={selectedLayer} onChange={(event) => setSelectedLayer(Number(event.target.value))}>{[1, 2, 3, 4].map((layer) => <option value={layer} key={layer}>layer {layer}</option>)}</select><span>inspect row</span><select value={focusRow} onChange={(event) => setFocusRow(Number(event.target.value))}>{tokens.map((token, index) => <option value={index} key={token + index}>{index}: {token.trim()}</option>)}</select></div></div>
            <div className={styles.stageGrid}>
                <StageCard id="input" active={activeStage === "input"} title="Embedding lookup" subtitle="01 / input" onClick={() => setActiveStage("input")}>
                    <p className={styles.cardCopy}>The integer encoding selects one row from each table. Click a token above to follow another row.</p><div className={styles.encodingLine}><span>encoding</span><b>&quot;{tokens[focusRow].trim()}&quot;</b><b>id {tokenIds[focusRow]}</b><b>position {focusRow}</b></div><Vector values={tokenEmbedding[focusRow]} label="E token" /><Vector values={positionEmbedding[focusRow]} label="E pos" /><Vector values={pass.embedded[focusRow]} label="sum x" /><MatrixTable matrix={pass.embedded} rowLabels={rowLabels} colLabels={["d0", "d1", "d2", "d3"]} focusRow={focusRow} /><Formula>x[t] = E_token[id] + E_position[t]</Formula>
                </StageCard>

                <StageCard id="attention" active={activeStage === "attention"} title="Causal attention" subtitle="02 / context" onClick={() => setActiveStage("attention")}>
                    <p className={styles.cardCopy}>The selected row creates a query. It compares against every key it is allowed to see, then uses the weights to mix value vectors.</p><div className={styles.miniLabel}>attention weights for: {tokens[focusRow].trim()}</div><MatrixTable matrix={[layerState.attention[focusRow]]} colLabels={rowLabels} /><Vector values={layerState.queries[focusRow]} label="q" /><Vector values={layerState.keys[focusRow]} label="k" /><Vector values={layerState.values[focusRow]} label="v" /><Formula>softmax(QK^T / sqrt(d)) V</Formula>
                </StageCard>

                <StageCard id="block" active={activeStage === "block"} title="Residual + MLP" subtitle="03 / transform" onClick={() => setActiveStage("block")}>
                    <p className={styles.cardCopy}>Context is added back to the stream. A small feed-forward network then transforms each row independently.</p><div className={styles.miniLabel}>selected row before and after layer {selectedLayer}</div><Vector values={layerState.residual[focusRow]} label="residual" /><Vector values={layerState.ffHidden[focusRow]} label="relu" /><Vector values={layerState.blockOutput[focusRow]} label="output" /><Formula>h&apos; = (x + Attention(x)) + W2 ReLU(W1 x)</Formula>
                </StageCard>

                <StageCard id="prediction" active={activeStage === "prediction"} title="Logits -> probability" subtitle="04 / output" onClick={() => setActiveStage("prediction")}>
                    <p className={styles.cardCopy}>Only the final row is used for the next-token decision. The output projection gives one logit per vocabulary item.</p><div className={styles.miniLabel}>final hidden row</div><Vector values={focusVector} label="h" /><div className={styles.logitList}>{topPredictions.map((prediction) => <div className={styles.logitRow} key={prediction.token}><span>{prediction.token}</span><b>{(prediction.probability * 100).toFixed(1)}%</b><i><em style={{ width: `${prediction.probability * 100}%` }} /></i></div>)}</div><Formula>P(token) = softmax(logits / temperature)</Formula></StageCard>
            </div>
        </section>

        <section className={styles.traceSection}><div className={styles.traceIntro}><span className={styles.sectionLabel}>EXACT LAYER TRACE</span><h2>Layer {selectedLayer}: every operation in order</h2><p>Shapes use the teaching model: sequence length N = 6 and hidden width d = 4. The displayed values are condensed, but each cell is a real result of the formula beside it.</p></div><div className={styles.layerRail}>{[1, 2, 3, 4].map((layer) => <button key={layer} className={selectedLayer === layer ? styles.layerActive : ""} onClick={() => setSelectedLayer(layer)}><b>Layer {layer}</b><span>H{`^${layer}`} = 6 x 4</span></button>)}</div><div className={styles.traceGrid}><TraceStep number="01" title="Input state" formula={selectedLayer === 1 ? "H0 = E_token + E_position" : `H${selectedLayer - 1} = previous layer output`} shape="6 x 4" matrix={previousState} rowLabels={rowLabels} colLabels={["d0", "d1", "d2", "d3"]} /><TraceStep number="02" title="Query projection" formula="Q = H Wq" shape="6 x 4" matrix={layerState.queries} rowLabels={rowLabels} /><TraceStep number="03" title="Key projection" formula="K = H Wk" shape="6 x 4" matrix={layerState.keys} rowLabels={rowLabels} /><TraceStep number="04" title="Value projection" formula="V = H Wv" shape="6 x 4" matrix={layerState.values} rowLabels={rowLabels} /><TraceStep number="05" title="Scaled scores + causal mask" formula="S = mask(Q K^T / sqrt(4))" shape="6 x 6" matrix={layerState.rawScores} rowLabels={rowLabels} colLabels={rowLabels} masked /><TraceStep number="06" title="Attention weights" formula="A = softmax(S)" shape="6 x 6" matrix={layerState.attention} rowLabels={rowLabels} colLabels={rowLabels} /><TraceStep number="07" title="Context aggregation" formula="C = A V" shape="6 x 4" matrix={layerState.context} rowLabels={rowLabels} /><TraceStep number="08" title="Residual + MLP output" formula="H_next = (H + C) + W2 ReLU(W1(H + C))" shape="6 x 4" matrix={layerState.blockOutput} rowLabels={rowLabels} /></div><div className={styles.traceFooter}>Layer 4 output -&gt; <b>Wout</b> -&gt; logits [6 x 50,257] -&gt; softmax -&gt; next-token probability. The real GPT-2 panel above shows the actual checkpoint output for selected text.</div></section>

        <section className={styles.inspector}><div className={styles.inspectorIntro}><span className={styles.sectionLabel}>WEIGHT INSPECTOR</span><h2>Parameters are not decoration.</h2><p>These matrices are declared in the tiny model source and are intentionally fixed teaching values, not weights downloaded from a trained nanoGPT checkpoint. Change the temperature to see only the final softmax change.</p><div className={styles.provenance}><b>NUMBER PROVENANCE</b><span>IDs: toy vocabulary lookup</span><span>E_token, E_position: fixed tables</span><span>Wq, Wk, Wv, W1, W2, Wout: fixed toy matrices</span><span>Everything after lookup: computed by matrix multiplication</span></div><label>temperature <input type="range" min="0.2" max="1.5" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /><b>{temperature.toFixed(1)}</b></label></div><div className={styles.parameterGrid}><ParameterPanel title="Wq" matrix={Wq} note="Fixed toy query projection: what does each token want to read?" /><ParameterPanel title="Wk" matrix={Wk} note="Fixed toy key projection: what information does each token offer?" /><ParameterPanel title="Wv" matrix={Wv} note="Fixed toy value projection: what content gets mixed?" /><ParameterPanel title="Wout" matrix={Wout} note="Fixed toy output projection: hidden features to vocabulary logits." /></div></section>

        <section className={styles.mathSection}><div><span className={styles.sectionLabel}>THE WHOLE STORY</span><h2>One line, many visible operations.</h2></div><div className={styles.mathChain}><span>tokens</span><b>-&gt;</b><span>X = E<sub>token</sub> + E<sub>position</sub></span><b>-&gt;</b><span>Q, K, V</span><b>-&gt;</b><span>masked softmax</span><b>-&gt;</b><span>residual + MLP</span><b>-&gt;</b><strong>next-token distribution</strong></div></section>
        <footer className={styles.footer}><span>nanoGPT 101 / transparent by design</span><span>Inspired by <a href="https://github.com/karpathy/nanoGPT" target="_blank" rel="noreferrer">karpathy/nanoGPT</a></span></footer>
    </main>;
}
