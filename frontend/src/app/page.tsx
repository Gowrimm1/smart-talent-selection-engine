"use client";
import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first!");
    setLoading(true);
    
    const formData = new FormData();
    formData.append('resume', file);
    formData.append('jd', ""); // This uses your backend's SAMPLE_JD

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Backend not responding. Is your Python server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-black text-white p-8 font-sans">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12 border-b border-gray-800 pb-6">
          <h1 className="text-4xl font-extrabold tracking-tight text-blue-500">
            Cymonic <span className="text-white">Talent Engine</span>
          </h1>
          <p className="text-gray-400 mt-2">Semantic AI Resume Ranking & Analysis</p>
        </header>

        {/* Upload Section */}
        <section className="bg-gray-900 p-8 rounded-2xl border border-gray-800 shadow-xl">
          <h2 className="text-xl font-semibold mb-4">Upload Candidate Resume</h2>
          <div className="flex flex-col gap-4">
            <input 
              type="file" 
              accept=".pdf,.docx"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer"
            />
            <button 
              onClick={handleUpload}
              disabled={loading}
              className={`py-3 px-6 rounded-xl font-bold transition-all ${loading ? 'bg-gray-700' : 'bg-blue-600 hover:bg-blue-500 hover:scale-[1.02]'}`}
            >
              {loading ? "AI is Analyzing..." : "Run Semantic Match"}
            </button>
          </div>
        </section>

        {/* Result Section */}
        {result && (
          <div className="mt-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className={`p-8 rounded-2xl border-l-8 bg-gray-900 border-gray-800`} style={{ borderColor: result.verdict_color }}>
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h3 className="text-3xl font-bold">{result.score}% Match</h3>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest`} style={{ backgroundColor: result.verdict_color, color: 'black' }}>
                    {result.verdict}
                  </span>
                </div>
                <div className="text-right text-gray-500 text-sm">
                  File: {result.filename}
                </div>
              </div>
              
              <p className="text-gray-300 leading-relaxed italic mb-6">"{result.summary}"</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-green-400 font-bold mb-2 uppercase text-xs">Top Strengths</h4>
                  <ul className="space-y-1">
                    {result.strengths.map((s: string) => <li key={s} className="text-gray-400 text-sm">• {s}</li>)}
                  </ul>
                </div>
                <div>
                  <h4 className="text-red-400 font-bold mb-2 uppercase text-xs">Potential Gaps</h4>
                  <ul className="space-y-1">
                    {result.gaps.map((g: string) => <li key={g} className="text-gray-400 text-sm">• {g}</li>)}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}