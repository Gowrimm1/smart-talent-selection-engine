"use client";

import { useState, useCallback, useRef } from "react";

// ── Types ──────────────────────────────────────────────────────────────────
type AnalyzeResult = {
  filename: string;
  extracted_text_preview: string;
  character_count: number;
  score: number;
  verdict: string;
  verdict_color: string;
  strengths: string[];
  gaps: string[];
  summary: string;
};

type RankItem = {
  rank: number;
  filename: string;
  score: number;
  verdict: string;
  verdict_color: string;
  strengths: string[];
  gaps: string[];
  summary: string;
};

type QuestionCategory = {
  category: string;
  icon: string;
  questions: string[];
};

type QuestionResult = {
  filename: string;
  candidate_name: string;
  total_questions: number;
  categories: QuestionCategory[];
};

// ── Score Ring Component ───────────────────────────────────────────────────
function ScoreRing({ score, color }: { score: number; color: string }) {
  const radius = 54;
  const circ = 2 * Math.PI * radius;
  const offset = circ - (score / 100) * circ;

  const ringColor =
    score >= 80 ? "#4ade80" : score >= 65 ? "#a3e635" : score >= 50 ? "#facc15" : score >= 35 ? "#fb923c" : "#f87171";

  return (
    <div className="relative w-36 h-36 flex items-center justify-center">
      <svg className="absolute top-0 left-0 -rotate-90" width="144" height="144">
        <circle cx="72" cy="72" r={radius} fill="none" stroke="#1e293b" strokeWidth="10" />
        <circle
          cx="72"
          cy="72"
          r={radius}
          fill="none"
          stroke={ringColor}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1s ease" }}
        />
      </svg>
      <div className="text-center z-10">
        <div className="text-3xl font-black text-white tracking-tight">{score}%</div>
        <div className="text-xs text-slate-400 font-medium mt-0.5 uppercase">Match</div>
      </div>
    </div>
  );
}

// ── Chip Component ─────────────────────────────────────────────────────────
function Chip({ label, variant }: { label: string; variant: "strength" | "gap" }) {
  return (
    <span
      className={`inline-block px-2.5 py-1 rounded-full text-xs font-semibold tracking-wide uppercase mr-1.5 mb-1.5 ${
        variant === "strength"
          ? "bg-emerald-900/60 text-emerald-300 border border-emerald-700/50"
          : "bg-rose-900/60 text-rose-300 border border-rose-700/50"
      }`}
    >
      {label}
    </span>
  );
}

// ── Result Card ────────────────────────────────────────────────────────────
function ResultCard({ result, rank }: { result: AnalyzeResult | RankItem; rank?: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="relative bg-slate-900/80 border border-slate-700/60 rounded-2xl p-6 backdrop-blur-sm hover:border-slate-500/70 transition-all duration-300 mb-4">
      {rank && (
        <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-xs font-black text-white shadow-lg shadow-indigo-900/50">
          #{rank}
        </div>
      )}
      <div className="flex items-start gap-6 flex-col md:flex-row">
        <div className="mx-auto md:mx-0">
          <ScoreRing score={result.score} color={result.verdict_color} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-3 mb-1">
            <h3 className="text-white font-bold text-lg truncate">{result.filename}</h3>
            <span
              className="shrink-0 text-xs font-bold px-3 py-1 rounded-full"
              style={{
                background: result.score >= 80 ? "#14532d" : result.score >= 50 ? "#713f12" : "#450a0a",
                color: result.score >= 80 ? "#4ade80" : result.score >= 50 ? "#fbbf24" : "#f87171",
              }}
            >
              {result.verdict}
            </span>
          </div>
          <p className="text-slate-400 text-sm leading-relaxed mb-3">{result.summary}</p>
          <div className="flex flex-wrap mb-1">
            {result.strengths.slice(0, 8).map((s) => (
              <Chip key={s} label={s} variant="strength" />
            ))}
            {result.gaps.slice(0, 8).map((g) => (
              <Chip key={g} label={g} variant="gap" />
            ))}
          </div>
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-indigo-400 text-xs font-semibold hover:text-indigo-300 transition-colors mt-2"
          >
            {expanded ? "▲ Hide Preview" : "▼ Show Text Preview"}
          </button>
          {expanded && "extracted_text_preview" in result && (
            <pre className="mt-3 p-3 bg-slate-950/80 rounded-lg text-xs text-slate-400 font-mono whitespace-pre-wrap max-h-40 overflow-y-auto border border-slate-800">
              {result.extracted_text_preview}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Drop Zone ──────────────────────────────────────────────────────────────
function DropZone({ onFiles, multi }: { onFiles: (files: File[]) => void; multi: boolean }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const handle = (files: FileList | null) => {
    if (!files) return;
    onFiles(Array.from(files).filter((f) => /\.(pdf|docx?)$/i.test(f.name)));
  };
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files); }}
      onClick={() => inputRef.current?.click()}
      className={`cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center py-10 px-6 text-center h-full ${
        dragging ? "border-indigo-400 bg-indigo-950/40" : "border-slate-700 bg-slate-900/40 hover:border-slate-500"
      }`}
    >
      <input ref={inputRef} type="file" accept=".pdf,.docx,.doc" multiple={multi} className="hidden" onChange={(e) => handle(e.target.files)} />
      <div className="text-4xl mb-3">📄</div>
      <p className="text-slate-300 font-semibold">{multi ? "Drop multiple resumes" : "Drop single resume"}</p>
      <p className="text-slate-500 text-sm mt-1">PDF or DOCX allowed</p>
    </div>
  );
}

const SAMPLE_JD = `Senior Machine Learning Engineer...`; // Shortened for display

export default function Home() {
  const [mode, setMode] = useState<"single" | "rank" | "questions">("single");
  const [jd, setJd] = useState(SAMPLE_JD);
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [singleResult, setSingleResult] = useState<AnalyzeResult | null>(null);
  const [rankResults, setRankResults] = useState<RankItem[] | null>(null);
  const [questionResult, setQuestionResult] = useState<QuestionResult | null>(null);

  const reset = () => { setFiles([]); setSingleResult(null); setRankResults(null); setQuestionResult(null); setError(null); };

  const submit = useCallback(async () => {
    if (!files.length) return;
    setLoading(true); setError(null);
    const API = "http://localhost:8000";
    try {
      const fd = new FormData();
      fd.append("jd", jd);
      if (mode === "single") {
        fd.append("resume", files[0]);
        const res = await fetch(`${API}/analyze`, { method: "POST", body: fd });
        setSingleResult(await res.json());
      } else if (mode === "rank") {
        files.forEach((f) => fd.append("resumes", f));
        const res = await fetch(`${API}/rank`, { method: "POST", body: fd });
        const data = await res.json();
        setRankResults(data.results);
      } else {
        fd.append("resume", files[0]);
        const res = await fetch(`${API}/generate-questions`, { method: "POST", body: fd });
        setQuestionResult(await res.json());
      }
    } catch (e) { setError("Backend Connection Error. Is your Python server running?"); } finally { setLoading(false); }
  }, [files, jd, mode]);

  return (
    <main className="min-h-screen bg-[#050508] text-white font-sans">
      <div className="max-w-5xl mx-auto p-6 md:p-10">
        <header className="mb-12 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-indigo-400 mb-2">TalentRank AI</h1>
          <p className="text-slate-400">Semantic Resume Analysis Engine</p>
        </header>

        <nav className="flex justify-center gap-2 mb-8 bg-slate-900/50 p-1 rounded-xl w-fit mx-auto border border-slate-800">
          {["single", "rank", "questions"].map((m) => (
            <button 
              key={m} 
              onClick={() => { setMode(m as any); reset(); }} 
              className={`px-6 py-2 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${mode === m ? "bg-indigo-600 text-white shadow-lg" : "text-slate-400 hover:text-white"}`}
            >
              {m === "single" ? "Analyze" : m === "rank" ? "Rank" : "Questions"}
            </button>
          ))}
        </nav>

        {!singleResult && !rankResults && !questionResult ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
            <div className="flex flex-col">
              <label className="text-sm font-bold text-slate-400 mb-2 uppercase">Job Description</label>
              <textarea 
                value={jd} 
                onChange={(e) => setJd(e.target.value)} 
                className="flex-1 min-h-[300px] bg-slate-900 border border-slate-700 rounded-2xl p-4 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                placeholder="Paste Job Description here..."
              />
            </div>
            <div className="flex flex-col">
              <label className="text-sm font-bold text-slate-400 mb-2 uppercase">Resume Upload</label>
              <DropZone multi={mode === "rank"} onFiles={setFiles} />
            </div>
            <button 
              onClick={submit} 
              disabled={loading || !files.length}
              className="md:col-span-2 py-4 rounded-2xl bg-indigo-600 font-bold hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-xl shadow-indigo-900/20"
            >
              {loading ? "AI is processing..." : "Run AI Engine"}
            </button>
            {error && <p className="md:col-span-2 text-center text-red-400 text-sm">{error}</p>}
          </div>
        ) : (
          <div className="animate-in fade-in duration-700">
            <button onClick={reset} className="mb-8 text-indigo-400 font-bold hover:underline">← Start New Session</button>
            {singleResult && <ResultCard result={singleResult} />}
            {rankResults && rankResults.map((r) => <ResultCard key={r.rank} result={r} rank={r.rank} />)}
            {questionResult && (
              <div className="bg-slate-900 p-8 rounded-3xl border border-indigo-500/20">
                <h2 className="text-2xl font-bold mb-6">Suggested Interview Questions</h2>
                {questionResult.categories.map(cat => (
                  <div key={cat.category} className="mb-6">
                    <h3 className="flex items-center gap-2 text-indigo-300 font-bold mb-3">{cat.icon} {cat.category}</h3>
                    <ul className="space-y-2">
                      {cat.questions.map((q, i) => (
                        <li key={i} className="p-3 bg-black/40 rounded-xl border border-white/5 text-slate-300 text-sm">{q}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}