"use client";

import { FormEvent, useState } from "react";

type Source = {
  title: string;
  state_or_ministry: string;
  section: string;
  source_file: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};
function formatAnswer(text: string) {
  return text.split("\n").map((line, lineIndex) => {
    const parts = line.split(/(\[Source \d+(?:,\s*Source \d+)*\])/g);

    return (
      <div key={lineIndex} className="min-h-6">
        {parts.map((part, index) => {
          const match = part.match(
            /^\[Source (\d+(?:,\s*Source \d+)*)\]$/
          );

          if (!match) {
            return <span key={index}>{part}</span>;
          }

          return (
            <span
              key={index}
              className="ml-1 inline-flex rounded-md bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600"
            >
              Source {match[1]}
            </span>
          );
        })}
      </div>
    );
  });
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  async function askQuestion(

  event: FormEvent,

  suggestedQuestion?: string

)  {
    event.preventDefault();

    const query = (suggestedQuestion ?? question).trim();

    if (!query || loading) return;

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: query,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: query,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get a response from the server.");
      }

      const data = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
        },
      ]);
    } catch (error) {
      console.error(error);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "Sorry, I couldn't connect to the SehajAI server. Please make sure the FastAPI backend is running.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    // <main className="min-h-screen bg-gray-50">
    <main className="h-screen overflow-hidden bg-gray-50">
      <div className="mx-auto flex h-full max-w-4xl flex-col">
        {/* Header */}
        <header className="flex items-center justify-between border-b bg-white px-6 py-5">
  <div>
    <h1 className="text-2xl font-bold text-gray-900">SehajAI</h1>
    <p className="mt-1 text-sm text-gray-500">
      Government scheme assistant
    </p>
  </div>

  {messages.length > 0 && (
    <button
      type="button"
      onClick={() => setMessages([])}
      disabled={loading}
      className="rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-600 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
    >
      Clear chat
    </button>
  )}
</header>

        {/* Chat */}
        <section className="min-h-0 flex-1 space-y-6 overflow-y-auto px-6 py-8">
          {messages.length === 0 && (
  <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
    <h2 className="text-3xl font-semibold text-gray-900">
      How can I help you?
    </h2>

    <p className="mt-3 max-w-lg text-gray-500">
      Ask about government schemes, eligibility, benefits, documents,
      or application procedures.
    </p>

    <div className="mt-8 grid w-full max-w-2xl gap-3 sm:grid-cols-3">
      {[
        "What documents are required for the Lakshadweep Scholarship Scheme?",
        "Who is eligible for the 25% Capital Investment Subsidy?",
        "What government schemes are available for farmers?",
      ].map((suggestion) => (
        <button
          key={suggestion}
          type="button"
          onClick={() => {
  askQuestion(
    { preventDefault: () => {} } as FormEvent,
    suggestion
  );
}}
          className="rounded-xl border border-gray-200 bg-white p-4 text-left text-sm text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50"
        >
          {suggestion}
        </button>
      ))}
    </div>
  </div>
)}

          {messages.map((message, index) => (
            <div
              key={index}
              className={
                message.role === "user"
                  ? "flex justify-end"
                  : "flex justify-start"
              }
            >
              <div
                className={
                  message.role === "user"
                    ? "max-w-[80%] rounded-2xl bg-gray-900 px-5 py-3 text-white"
                    : "max-w-[85%] rounded-2xl bg-white px-5 py-4 text-gray-900 shadow-sm"
                }
              >
                <div className="text-sm leading-6">
  {formatAnswer(message.content)}
</div>

                {message.sources && message.sources.length > 0 && (
  <div className="mt-5 border-t pt-4">
    <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-gray-500">
      Sources
    </p>

    <div className="space-y-2">
      {message.sources.map((source, sourceIndex) => (
        <div
          key={sourceIndex}
          className="rounded-xl border border-gray-200 bg-gray-50 p-3 transition hover:bg-gray-100"
        >
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-white text-xs font-semibold text-gray-500 shadow-sm">
              {sourceIndex + 1}
            </div>

            <div className="min-w-0">
              <p className="text-sm font-medium text-gray-900">
                {source.title}
              </p>

              <p className="mt-1 text-xs text-gray-500">
                {source.state_or_ministry}
              </p>

              <p className="mt-1 text-xs text-gray-400">
                Section: {source.section}
              </p>
            </div>
          </div>
        </div>
      ))}
    </div>
  </div>
)}
              </div>
            </div>
          ))}

          {loading && (
  <div className="flex justify-start">
    <div className="rounded-2xl bg-white px-5 py-4 text-sm text-gray-500 shadow-sm">
      <div className="flex items-center gap-2">
        <span className="flex gap-1">
          <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.3s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.15s]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400" />
        </span>
        <span>Searching government schemes...</span>
      </div>
    </div>
  </div>
)}
        </section>

        {/* Input */}
        <form
          onSubmit={askQuestion}
          className="border-t bg-white p-4"
        >
          <div className="flex gap-3">
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about a government scheme..."
              disabled={loading}
              className="flex-1 rounded-xl border border-gray-300 px-4 py-3 text-sm outline-none focus:border-gray-500"
            />

            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="rounded-xl bg-gray-900 px-6 py-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "..." : "Ask"}
            </button>
          </div>
        </form>
      </div>
    </main>
  );
}