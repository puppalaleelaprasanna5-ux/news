import { useRef, useState } from "react";
import { Loader2, RotateCcw, ShieldCheck, X } from "lucide-react";
import { MAX_ARTICLE_LENGTH } from "@/lib/api";

interface ArticleInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isSubmitting: boolean;
}

const EXAMPLES: Array<{ title: string; text: string }> = [
  {
    title: "Local news report",
    text: "The city council voted 7-2 on Tuesday to approve the riverside park project, ending a two-year consultation process. Construction is expected to begin in the spring, with the first phase opening to the public by autumn. Council members cited community surveys showing broad support for expanded green space along the waterfront.",
  },
  {
    title: "Viral claim",
    text: "SHOCKING: Doctors HATE this one simple trick that instantly cures the common cold — banned by hospitals because it works too well! Big Pharma doesn't want you to know this secret remedy that has been suppressed for decades.",
  },
];

export function ArticleInput({ value, onChange, onSubmit, isSubmitting }: ArticleInputProps) {
  const [showExamples, setShowExamples] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const charCount = value.length;
  const canSubmit = value.trim().length > 0 && !isSubmitting;

  return (
    <section
      aria-label="Article input"
      className="rounded-xl border border-border bg-card p-5 sm:p-6"
    >
      <div className="flex items-baseline justify-between gap-4">
        <label
          htmlFor="article-input"
          className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground"
        >
          Paste your article
        </label>
        <span aria-live="polite" className="text-xs tabular-nums text-muted-foreground">
          {charCount.toLocaleString()} / {MAX_ARTICLE_LENGTH.toLocaleString()}
        </span>
      </div>

      <div className="relative mt-2">
        <textarea
          ref={textareaRef}
          id="article-input"
          value={value}
          maxLength={MAX_ARTICLE_LENGTH}
          onChange={(event) => onChange(event.target.value)}
          placeholder="Paste a headline or article to analyze..."
          rows={7}
          className="w-full resize-y rounded-lg border border-input bg-background px-4 py-3 text-[15px] leading-relaxed text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ring"
        />
        {value.length > 0 && (
          <button
            type="button"
            onClick={() => onChange("")}
            className="absolute right-2 top-2 flex items-center gap-1 rounded-md border border-border bg-card px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <X className="h-3.5 w-3.5" aria-hidden="true" />
            Clear
          </button>
        )}
      </div>

      <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-muted-foreground">
          For better context, paste the full article rather than just a headline.
        </p>
        <button
          type="button"
          onClick={onSubmit}
          disabled={!canSubmit}
          className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              Analyzing…
            </>
          ) : (
            "Analyze article"
          )}
        </button>
      </div>

      <div className="mt-4 border-t border-border pt-3">
        {!showExamples ? (
          <button
            type="button"
            onClick={() => setShowExamples(true)}
            className="text-sm text-muted-foreground underline decoration-border underline-offset-4 transition-colors hover:text-primary"
          >
            Try an example
          </button>
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm text-muted-foreground">Examples:</span>
            {EXAMPLES.map((example) => (
              <button
                key={example.title}
                type="button"
                onClick={() => {
                  onChange(example.text);
                  setShowExamples(false);
                }}
                className="rounded-md border border-border bg-secondary px-2.5 py-1 text-sm text-secondary-foreground transition-colors hover:bg-accent"
              >
                {example.title}
              </button>
            ))}
            <button
              type="button"
              onClick={() => setShowExamples(false)}
              className="text-sm text-muted-foreground hover:text-foreground"
            >
              Hide
            </button>
          </div>
        )}
      </div>
    </section>
  );
}

export function TrustNote() {
  return (
    <p className="mt-4 flex items-start gap-2 text-sm text-muted-foreground">
      <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-primary/70" aria-hidden="true" />
      <span>
        AI-assisted analysis, not a final fact-check. Always verify important claims with reliable
        sources.
      </span>
    </p>
  );
}

export function RetryButton({ onRetry }: { onRetry: () => void }) {
  return (
    <button
      type="button"
      onClick={onRetry}
      className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
    >
      <RotateCcw className="h-4 w-4" aria-hidden="true" />
      Try again
    </button>
  );
}
