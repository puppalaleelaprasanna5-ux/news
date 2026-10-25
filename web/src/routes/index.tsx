import { useCallback, useEffect, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { ArticleInput, TrustNote } from "@/components/ArticleInput";
import { ResultsSection } from "@/components/AnalysisResults";
import { ApiError, analyzeArticle, type AnalysisResult } from "@/lib/api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "NewsCred — Before you believe it, check it." },
      {
        name: "description",
        content:
          "NewsCred analyzes a news article with an AI classifier and evidence retrieval, so you can see what a model predicts and what reference documents say.",
      },
      { property: "og:title", content: "NewsCred — Before you believe it, check it." },
      {
        property: "og:description",
        content:
          "Explore what an AI classifier predicts and what your available reference documents say about an article.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

type SubmitState = "idle" | "loading" | "error" | "done";

function Index() {
  const [article, setArticle] = useState("");
  const [submitState, setSubmitState] = useState<SubmitState>("idle");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const lastArticleRef = useRef<string>("");

  const runAnalysis = useCallback(async (text: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    lastArticleRef.current = text;
    setSubmitState("loading");
    setError(null);
    try {
      const analysis = await analyzeArticle(text, controller.signal);
      setResult(analysis);
      setSubmitState("done");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong while analyzing the article. Please try again.",
      );
      setSubmitState("error");
    }
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const handleSubmit = useCallback(() => {
    if (article.trim().length === 0) return;
    void runAnalysis(article);
  }, [article, runAnalysis]);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="w-full border-b border-border bg-background">
        <div className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-4 px-4 py-4 sm:px-6 sm:flex sm:flex-wrap sm:justify-between lg:px-[5.5%]">
          <div>
            <p className="font-display text-2xl font-semibold leading-none text-foreground">
              NewsCred<span className="text-primary">.</span>
            </p>
            <p className="mt-1 text-sm text-muted-foreground">News analysis, with evidence.</p>
          </div>
        </div>
      </header>

      <main className="w-full flex-1 px-4 pb-16 pt-10 sm:px-6 sm:pb-24 sm:pt-14 lg:px-[5.5%]">
        <section className="mx-auto max-w-3xl text-center lg:max-w-none">
          <h1 className="font-display text-4xl font-semibold tracking-tight text-foreground sm:text-5xl">
            Before you believe it, check it.
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-[15px] leading-relaxed text-muted-foreground sm:max-w-2xl">
            Explore what an AI classifier predicts and what your available reference documents say
            about an article.
          </p>
        </section>

        <section className="mt-8 sm:mt-10">
          <ArticleInput
            value={article}
            onChange={setArticle}
            onSubmit={handleSubmit}
            isSubmitting={submitState === "loading"}
          />
          <TrustNote />
          <ResultsSection
            state={submitState}
            result={result}
            error={error}
            onRetry={() => void runAnalysis(lastArticleRef.current)}
          />
        </section>
      </main>
    </div>
  );
}
