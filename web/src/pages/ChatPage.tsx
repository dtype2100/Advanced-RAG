import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { apiJson } from "../api";
import "./ChatPage.css";

type ChatResponse = {
  question: string;
  answer: string;
  sources: string[];
  retries: number;
  clarification_needed?: boolean;
  clarification_question?: string | null;
};

type Msg =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string; sources?: string[]; retries?: number; clarify?: string };

export function ChatPage() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openSources, setOpenSources] = useState<number | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  async function send() {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setError(null);
    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await apiJson<ChatResponse>("/api/v1/query", {
        method: "POST",
        body: JSON.stringify({ question: q, top_k: 5 }),
      });
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          retries: res.retries,
          clarify: res.clarification_needed ? res.clarification_question || undefined : undefined,
        },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-page">
      <div className="chat-scroll">
        {messages.length === 0 && (
          <p style={{ color: "var(--muted)", textAlign: "center", marginTop: "2rem" }}>
            질문을 입력하세요. 문서는 API로 먼저 색인해 두어야 답변이 풍부해집니다.
          </p>
        )}
        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <div key={i} className="msg user">
              <div className="msg-meta">You</div>
              <div className="msg-body">{msg.content}</div>
            </div>
          ) : (
            <div key={i} className="msg assistant">
              <div className="msg-meta">Assistant{msg.retries ? ` · retries ${msg.retries}` : ""}</div>
              {msg.clarify && <div className="clarify-banner">Clarification: {msg.clarify}</div>}
              <div className="msg-body">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <>
                  <button
                    type="button"
                    className="sources-toggle"
                    onClick={() => setOpenSources(openSources === i ? null : i)}
                  >
                    {openSources === i ? "Hide sources" : `Sources (${msg.sources.length})`}
                  </button>
                  {openSources === i && (
                    <div className="sources-panel">
                      {msg.sources.map((s, j) => (
                        <div key={j} style={{ marginBottom: "0.5rem" }}>
                          <strong>[{j + 1}]</strong> {s.slice(0, 2000)}
                          {s.length > 2000 ? "…" : ""}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          ),
        )}
        <div ref={bottomRef} />
      </div>
      {error && <div className="error-text">{error}</div>}
      <div className="chat-input-row">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          placeholder="메시지를 입력… (Enter 전송, Shift+Enter 줄바꿈)"
          rows={2}
        />
        <button type="button" disabled={loading || !input.trim()} onClick={() => void send()}>
          {loading ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
