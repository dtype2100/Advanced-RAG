import { useCallback, useEffect, useState } from "react";
import { apiJson } from "../api";
import "./StudioPage.css";

type RuntimeState = {
  overrides: Record<string, unknown>;
  effective: Record<string, unknown>;
};

type ReadonlyCfg = Record<string, unknown>;

export function StudioPage() {
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [config, setConfig] = useState<ReadonlyCfg | null>(null);
  const [probe, setProbe] = useState<Record<string, unknown> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [mq, setMq] = useState(false);
  const [topK, setTopK] = useState("");
  const [ground, setGround] = useState("");
  const [maxRetry, setMaxRetry] = useState("");
  const [rerankK, setRerankK] = useState("");

  const load = useCallback(async () => {
    setErr(null);
    try {
      const [rt, cf] = await Promise.all([
        apiJson<RuntimeState>("/api/v1/studio/runtime"),
        apiJson<ReadonlyCfg>("/api/v1/studio/config"),
      ]);
      setRuntime(rt);
      setConfig(cf);
      const eff = rt.effective as Record<string, unknown>;
      setMq(Boolean(eff.multi_query));
      setTopK(String(eff.max_retrieval_docs ?? ""));
      setGround(String(eff.grounding_threshold ?? ""));
      setMaxRetry(String(eff.max_retries ?? ""));
      const rk = eff.rerank_top_k;
      setRerankK(rk == null ? "" : String(rk));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function saveRuntime() {
    setErr(null);
    const body: Record<string, unknown> = {
      multi_query: mq,
      max_retrieval_docs: topK === "" ? null : parseInt(topK, 10),
      grounding_threshold: ground === "" ? null : parseFloat(ground),
      max_retries: maxRetry === "" ? null : parseInt(maxRetry, 10),
      rerank_top_k: rerankK === "" ? null : parseInt(rerankK, 10),
    };
    try {
      const rt = await apiJson<RuntimeState>("/api/v1/studio/runtime", {
        method: "PATCH",
        body: JSON.stringify(body),
      });
      setRuntime(rt);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function clearOverrides() {
    setErr(null);
    try {
      const rt = await apiJson<RuntimeState>("/api/v1/studio/runtime", {
        method: "PATCH",
        body: JSON.stringify({
          multi_query: null,
          max_retrieval_docs: null,
          grounding_threshold: null,
          max_retries: null,
          rerank_top_k: null,
        }),
      });
      setRuntime(rt);
      await load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function runProbe() {
    setErr(null);
    try {
      const p = await apiJson<Record<string, unknown>>("/api/v1/studio/probe", {
        method: "POST",
      });
      setProbe(p);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="studio-page">
      <h1>Studio</h1>
      <p className="muted">
        런타임 값은 이 API 프로세스에만 적용됩니다. vLLM·Qdrant URL 등은 .env로 두고 재시작하세요.
      </p>
      {err && <p className="muted" style={{ color: "#e85d5d" }}>{err}</p>}

      <section className="studio-section">
        <h2>런타임 RAG 파라미터</h2>
        <div className="grid-2">
          <div className="field">
            <label>
              <input type="checkbox" checked={mq} onChange={(e) => setMq(e.target.checked)} />
              Multi-query retrieval
            </label>
          </div>
          <div className="field">
            <label>Max retrieval docs (top_k)</label>
            <input type="number" min={1} max={100} value={topK} onChange={(e) => setTopK(e.target.value)} />
          </div>
          <div className="field">
            <label>Grounding threshold (0–1, below → retry)</label>
            <input type="text" inputMode="decimal" value={ground} onChange={(e) => setGround(e.target.value)} />
          </div>
          <div className="field">
            <label>Max graph retries</label>
            <input type="number" min={0} max={20} value={maxRetry} onChange={(e) => setMaxRetry(e.target.value)} />
          </div>
          <div className="field">
            <label>Rerank top_k (빈 칸 = 전체 유지)</label>
            <input type="number" min={1} max={50} value={rerankK} onChange={(e) => setRerankK(e.target.value)} />
          </div>
        </div>
        <div className="studio-actions">
          <button type="button" className="primary" onClick={() => void saveRuntime()}>
            적용
          </button>
          <button type="button" onClick={() => void clearOverrides()}>
            오버라이드 초기화
          </button>
          <button type="button" onClick={() => void load()}>
            새로고침
          </button>
        </div>
        {runtime && (
          <>
            <p className="muted" style={{ marginTop: "0.75rem" }}>
              Effective (현재 해석값)
            </p>
            <pre className="pre-block">{JSON.stringify(runtime.effective, null, 2)}</pre>
          </>
        )}
      </section>

      <section className="studio-section">
        <h2>배포 설정 (읽기 전용)</h2>
        <button type="button" onClick={() => void load()}>
          다시 불러오기
        </button>
        {config && <pre className="pre-block" style={{ marginTop: "0.75rem" }}>{JSON.stringify(config, null, 2)}</pre>}
      </section>

      <section className="studio-section">
        <h2>연결 프로브</h2>
        <button type="button" className="primary" onClick={() => void runProbe()}>
          vLLM / Qdrant / TEI health 확인
        </button>
        {probe && <pre className="pre-block" style={{ marginTop: "0.75rem" }}>{JSON.stringify(probe, null, 2)}</pre>}
      </section>
    </div>
  );
}
