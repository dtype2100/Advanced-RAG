import { useCallback, useEffect, useState } from "react";
import { apiJson } from "../api";
import "./StudioPage.css";

type ReadonlyCfg = Record<string, unknown>;

type ProbeResult = {
  llm: string;
  qdrant: Record<string, unknown>;
};

export function StudioPage() {
  const [config, setConfig] = useState<ReadonlyCfg | null>(null);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setErr(null);
    try {
      const cf = await apiJson<ReadonlyCfg>("/api/v1/studio/config");
      setConfig(cf);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function runProbe() {
    setErr(null);
    try {
      const p = await apiJson<ProbeResult>("/api/v1/studio/probe", { method: "POST" });
      setProbe(p);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="studio-page">
      <h1>Studio</h1>
      <p className="muted">
        배포 설정은 .env / 컨테이너 환경 변수로 관리합니다. 변경 후 API를 재시작하세요.
      </p>
      {err && <p className="muted" style={{ color: "#e85d5d" }}>{err}</p>}

      <section className="studio-section">
        <h2>배포 설정 (읽기 전용)</h2>
        <button type="button" onClick={() => void load()}>
          다시 불러오기
        </button>
        {config && (
          <pre className="pre-block" style={{ marginTop: "0.75rem" }}>
            {JSON.stringify(config, null, 2)}
          </pre>
        )}
      </section>

      <section className="studio-section">
        <h2>연결 프로브</h2>
        <button type="button" className="primary" onClick={() => void runProbe()}>
          LLM / Qdrant health 확인
        </button>
        {probe && (
          <pre className="pre-block" style={{ marginTop: "0.75rem" }}>
            {JSON.stringify(probe, null, 2)}
          </pre>
        )}
      </section>
    </div>
  );
}
