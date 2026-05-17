import { useEffect, useRef, useState } from "react";

const REDDIT_ORANGE = "#FF4500";
const REDDIT_ORANGE_LIGHT = "#FF7A4D";
const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:8001";

const RedditLogo = ({ size = 30 }) => (
    <svg width={size} height={size} viewBox="0 0 20 20">
        <circle cx="10" cy="10" r="10" fill={REDDIT_ORANGE} />
        <path d="M16.67 10a1.46 1.46 0 00-2.47-1 7.12 7.12 0 00-3.85-1.23l.65-3.07 2.12.45a1 1 0 101.07-1 1 1 0 00-.95.68l-2.38-.5a.15.15 0 00-.18.11l-.73 3.44a7.14 7.14 0 00-3.89 1.23 1.46 1.46 0 10-1.61 2.39 2.89 2.89 0 000 .44c0 2.24 2.61 4.06 5.83 4.06s5.83-1.82 5.83-4.06a2.89 2.89 0 000-.44 1.46 1.46 0 00.49-1.5zm-9.6 1.33a1 1 0 111 1 1 1 0 01-1-1zm5.56 2.64a3.47 3.47 0 01-2.63.82 3.47 3.47 0 01-2.63-.82.2.2 0 01.28-.28 3.09 3.09 0 002.35.66 3.09 3.09 0 002.35-.66.2.2 0 01.28.28zm-.2-1.64a1 1 0 111-1 1 1 0 01-1 1z" fill="white" stroke="none" />
    </svg>
);

function Background() {
    return (
        <div style={{ position: "fixed", inset: 0, overflow: "hidden", zIndex: 0, pointerEvents: "none" }}>
            <div className="orb orb1" />
            <div className="orb orb2" />
            <div className="orb orb3" />
            <div className="orb orb4" />
        </div>
    );
}

const modeOptions = [
    {
        value: "subreddit",
        label: "Analyze a subreddit",
        description: "Fetch posts from a subreddit and score their sentiment.",
        placeholder: "technology",
    },
    {
        value: "post_comments",
        label: "Analyze comments in a specific post",
        description: "Pull comments from a Reddit post and inspect the tone.",
        placeholder: "https://www.reddit.com/r/.../comments/...",
    },
    {
        value: "compare",
        label: "Compare multiple subreddits",
        description: "Compare how several communities feel side by side.",
        placeholder: "technology, programming, science",
    },
    {
        value: "custom_text",
        label: "Custom text sentiment analysis",
        description: "Paste any text or multiple lines for instant analysis.",
        placeholder: "Paste your text here",
    },
];

const modeByValue = Object.fromEntries(modeOptions.map((mode) => [mode.value, mode]));

const sentimentPalette = {
    Positive: {
        color: "#6EE7B7",
        soft: "rgba(110,231,183,0.12)",
        border: "rgba(110,231,183,0.28)",
    },
    Neutral: {
        color: "#C4C4D4",
        soft: "rgba(196,196,212,0.10)",
        border: "rgba(196,196,212,0.22)",
    },
    Negative: {
        color: "#FCA5A5",
        soft: "rgba(252,165,165,0.12)",
        border: "rgba(252,165,165,0.28)",
    },
};

const defaultForm = {
    mode: "subreddit",
    subreddit: "technology",
    postUrl: "",
    compareText: "technology, programming, science",
    customText: "This product is surprisingly good, but the documentation is weak.",
    sortBy: "hot",
    limit: "20",
    commentLimit: "40",
};

function clamp(value, min, max, fallback) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isNaN(parsed)) return fallback;
    return Math.max(min, Math.min(max, parsed));
}

function formatCount(value) {
    if (value === null || value === undefined) return "0";
    const number = Number(value);
    if (Number.isNaN(number)) return String(value);
    if (Math.abs(number) >= 1000000) return `${(number / 1000000).toFixed(1)}M`;
    if (Math.abs(number) >= 1000) return `${(number / 1000).toFixed(1)}K`;
    return `${number}`;
}

function formatScore(value) {
    const number = Number(value ?? 0);
    return `${number > 0 ? "+" : ""}${number.toFixed(2)}`;
}

function formatPercent(value) {
    return `${Math.round(Number(value ?? 0))}%`;
}

function formatRelativeTime(seconds) {
    if (!seconds) return "recent";
    const delta = Date.now() / 1000 - Number(seconds);
    if (delta < 3600) return `${Math.max(1, Math.round(delta / 60))}m ago`;
    if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
    return `${Math.round(delta / 86400)}d ago`;
}

function splitCommaList(value) {
    return value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => item.replace(/^r\//i, ""));
}

function splitCustomText(value) {
    return value
        .split(/\n+/)
        .map((item) => item.trim())
        .filter(Boolean);
}

function sentimentLabel(score) {
    if (score >= 0.6) return "Positive";
    if (score <= 0.4) return "Negative";
    return "Neutral";
}

function SentimentPill({ sentiment, score }) {
    const normalized = String(sentiment || "Neutral");
    const palette = sentimentPalette[normalized] || sentimentPalette.Neutral;
    return (
        <span
            style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 12px",
                borderRadius: 999,
                background: palette.soft,
                border: `1px solid ${palette.border}`,
                color: palette.color,
                fontSize: 12,
                fontWeight: 700,
                whiteSpace: "nowrap",
            }}
        >
            <span style={{ width: 7, height: 7, borderRadius: 999, background: palette.color, boxShadow: `0 0 10px ${palette.color}70` }} />
            {normalized}
            {typeof score === "number" ? <span style={{ opacity: 0.75, fontWeight: 600 }}>{formatScore(score)}</span> : null}
        </span>
    );
}

function MetricCard({ label, value, hint, accent }) {
    return (
        <div
            style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 22,
                padding: 20,
                backdropFilter: "blur(22px)",
                boxShadow: "0 16px 40px rgba(0,0,0,0.16)",
                position: "relative",
                overflow: "hidden",
            }}
        >
            <div style={{ position: "absolute", inset: "auto -24px -24px auto", width: 120, height: 120, borderRadius: 999, background: `radial-gradient(circle, ${accent}26, transparent 68%)`, pointerEvents: "none" }} />
            <div style={{ fontSize: 11, letterSpacing: 1.1, color: "rgba(255,255,255,0.34)", fontWeight: 800, textTransform: "uppercase", marginBottom: 10 }}>{label}</div>
            <div style={{ fontSize: 30, fontWeight: 800, letterSpacing: -1.2, color: accent, lineHeight: 1 }}>{value}</div>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.24)", marginTop: 8 }}>{hint}</div>
        </div>
    );
}

function GlassPanel({ children, style = {} }) {
    return (
        <div
            style={{
                background: "rgba(255,255,255,0.045)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 28,
                backdropFilter: "blur(28px) saturate(180%)",
                boxShadow: "0 20px 50px rgba(0,0,0,0.24)",
                position: "relative",
                overflow: "hidden",
                ...style,
            }}
        >
            {children}
        </div>
    );
}

function Donut({ summary }) {
    const positive = Number(summary?.positive ?? 0);
    const neutral = Number(summary?.neutral ?? 0);
    const negative = Number(summary?.negative ?? 0);
    const total = Math.max(positive + neutral + negative, 1);
    const radius = 46;
    const circumference = 2 * Math.PI * radius;
    const segments = [
        { value: positive, color: sentimentPalette.Positive.color },
        { value: neutral, color: sentimentPalette.Neutral.color },
        { value: negative, color: sentimentPalette.Negative.color },
    ];
    let offset = circumference * 0.25;

    return (
        <svg width="140" height="140" viewBox="0 0 140 140">
            <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="14" />
            {segments.map((segment, index) => {
                const length = (segment.value / total) * circumference;
                const element = (
                    <circle
                        key={index}
                        cx="70"
                        cy="70"
                        r={radius}
                        fill="none"
                        stroke={segment.color}
                        strokeWidth="14"
                        strokeDasharray={`${Math.max(0, length - 3)} ${circumference}`}
                        strokeDashoffset={offset}
                        strokeLinecap="round"
                        style={{ filter: `drop-shadow(0 0 7px ${segment.color}80)` }}
                    />
                );
                offset -= length;
                return element;
            })}
            <text x="70" y="66" textAnchor="middle" fill="white" fontSize="20" fontWeight="800" fontFamily="DM Sans, sans-serif">
                {Math.round((positive / total) * 100)}%
            </text>
            <text x="70" y="84" textAnchor="middle" fill="rgba(255,255,255,0.34)" fontSize="10" fontFamily="DM Sans, sans-serif">
                positive
            </text>
        </svg>
    );
}

function ResultCard({ item }) {
    const sentiment = String(item.sentiment || sentimentLabel(item.sentiment_score));
    const palette = sentimentPalette[sentiment] || sentimentPalette.Neutral;
    const subtitle = item.parent_post_title || item.subreddit || item.group || item.author || "Reddit";
    const score = Number(item.sentiment_score ?? item.score ?? 0);

    return (
        <div
            style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 22,
                padding: 18,
            }}
        >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", marginBottom: 12 }}>
                <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 800, color: "rgba(255,255,255,0.88)", lineHeight: 1.4 }}>{item.title || item.label || "Result"}</div>
                    <div style={{ fontSize: 11.5, color: "rgba(255,255,255,0.34)", marginTop: 4 }}>{subtitle}</div>
                </div>
                <SentimentPill sentiment={sentiment} score={score} />
            </div>

            {item.content ? (
                <p style={{ margin: "0 0 14px", color: "rgba(255,255,255,0.56)", lineHeight: 1.72, fontSize: 13.5 }}>{item.content}</p>
            ) : null}

            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <div style={{ flex: 1, height: 4, borderRadius: 999, background: "rgba(255,255,255,0.07)", overflow: "hidden" }}>
                    <div
                        style={{
                            width: `${Math.min(100, Math.max(0, ((score + 1) / 2) * 100))}%`,
                            height: "100%",
                            background: `linear-gradient(90deg, ${palette.color}70, ${palette.color})`,
                            boxShadow: `0 0 8px ${palette.color}66`,
                            borderRadius: 999,
                        }}
                    />
                </div>
                <span style={{ width: 56, textAlign: "right", fontSize: 12, fontFamily: "monospace", color: palette.color, fontWeight: 800 }}>{formatScore(score)}</span>
            </div>

            <div style={{ display: "flex", flexWrap: "wrap", gap: 14, fontSize: 12, color: "rgba(255,255,255,0.28)" }}>
                {item.author ? <span>Author: {item.author}</span> : null}
                {typeof item.upvotes === "number" ? <span>Upvotes: {formatCount(item.upvotes)}</span> : null}
                {typeof item.comments === "number" ? <span>Comments: {formatCount(item.comments)}</span> : null}
                {typeof item.replies === "number" ? <span>Replies: {formatCount(item.replies)}</span> : null}
                {typeof item.created_utc === "number" ? <span>{formatRelativeTime(item.created_utc)}</span> : null}
            </div>

            {item.url ? (
                <div style={{ marginTop: 12 }}>
                    <a href={item.url} target="_blank" rel="noreferrer" style={{ color: REDDIT_ORANGE_LIGHT, textDecoration: "none", fontSize: 12.5, fontWeight: 700 }}>
                        Open source
                    </a>
                </div>
            ) : null}
        </div>
    );
}

function ComparisonCard({ item }) {
    return (
        <div
            style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 22,
                padding: 18,
            }}
        >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, marginBottom: 14 }}>
                <div>
                    <div style={{ fontSize: 16, fontWeight: 800, color: "white" }}>r/{item.subreddit}</div>
                    <div style={{ fontSize: 12, color: "rgba(255,255,255,0.28)", marginTop: 4 }}>{item.post_count ?? 0} posts analyzed</div>
                </div>
                <SentimentPill sentiment={sentimentLabel(item.avg_sentiment)} score={item.avg_sentiment} />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 10 }}>
                <div style={{ borderRadius: 16, padding: 12, background: "rgba(255,255,255,0.03)" }}>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.28)", marginBottom: 6 }}>Average sentiment</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: REDDIT_ORANGE_LIGHT }}>{formatScore(item.avg_sentiment)}</div>
                </div>
                <div style={{ borderRadius: 16, padding: 12, background: "rgba(255,255,255,0.03)" }}>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.28)", marginBottom: 6 }}>Positive</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: sentimentPalette.Positive.color }}>{formatPercent(item.positive_percentage)}</div>
                </div>
                <div style={{ borderRadius: 16, padding: 12, background: "rgba(255,255,255,0.03)" }}>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.28)", marginBottom: 6 }}>Posts</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: "white" }}>{formatCount(item.post_count)}</div>
                </div>
            </div>
        </div>
    );
}

export default function App() {
    const [form, setForm] = useState(defaultForm);
    const [activeMode, setActiveMode] = useState(defaultForm.mode);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [result, setResult] = useState(null);
    const [dashboardVisible, setDashboardVisible] = useState(false);
    const [sentimentFilter, setSentimentFilter] = useState("all");
    const [listSort, setListSort] = useState("score");
    const dashboardRef = useRef(null);

    useEffect(() => {
        if (!dashboardVisible || !dashboardRef.current) return;
        dashboardRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }, [dashboardVisible]);

    const resultMode = result?.mode;

    useEffect(() => {
        if (!resultMode) return;
        setSentimentFilter("all");
        setListSort("score");
    }, [resultMode]);

    const summary = result?.summary || { total: 0, positive: 0, neutral: 0, negative: 0, avg_score: 0 };
    const items = Array.isArray(result?.items) ? result.items : [];
    const filteredItems = items.filter((item) => sentimentFilter === "all" || (item.sentiment || sentimentLabel(item.sentiment_score)) === sentimentFilter);
    const sortedItems = [...filteredItems].sort((left, right) => {
        const leftScore = Number(left.sentiment_score ?? 0);
        const rightScore = Number(right.sentiment_score ?? 0);
        const leftUpvotes = Number(left.upvotes ?? 0);
        const rightUpvotes = Number(right.upvotes ?? 0);
        const leftComments = Number(left.comments ?? left.replies ?? 0);
        const rightComments = Number(right.comments ?? right.replies ?? 0);
        const leftCreated = Number(left.created_utc ?? 0);
        const rightCreated = Number(right.created_utc ?? 0);

        if (listSort === "upvotes") return rightUpvotes - leftUpvotes;
        if (listSort === "activity") return rightComments - leftComments;
        if (listSort === "recent") return rightCreated - leftCreated;
        return rightScore - leftScore;
    });

    const comparisonItems = Array.isArray(result?.comparisons) ? result.comparisons : [];
    const keywordItems = Array.isArray(result?.top_keywords) ? result.top_keywords : [];
    const requestMode = result?.request?.mode || activeMode;

    const updateField = (field, value) => setForm((current) => ({ ...current, [field]: value }));

    const handleAnalyze = async () => {
        setError("");
        setLoading(true);

        const payload = { mode: activeMode };

        if (activeMode === "subreddit") {
            payload.subreddit = form.subreddit.trim();
            payload.limit = clamp(form.limit, 1, 100, 20);
            payload.sort_by = form.sortBy;
        }

        if (activeMode === "post_comments") {
            payload.post_url = form.postUrl.trim();
            payload.limit = clamp(form.commentLimit, 1, 100, 40);
        }

        if (activeMode === "compare") {
            payload.subreddits = splitCommaList(form.compareText);
            payload.limit_per_sub = clamp(form.limit, 1, 100, 20);
            payload.sort_by = form.sortBy;
        }

        if (activeMode === "custom_text") {
            payload.texts = splitCustomText(form.customText);
            payload.text = form.customText.trim();
        }

        try {
            const response = await fetch(`${API_BASE}/api/analyze`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok || data.ok === false) {
                throw new Error(data.error || `Request failed with status ${response.status}`);
            }

            setResult(data);
            setDashboardVisible(true);
        } catch (fetchError) {
            setError(fetchError.message || "Unable to connect to the backend.");
            setResult(null);
            setDashboardVisible(true);
        } finally {
            setLoading(false);
        }
    };

    const handleModeChange = (mode) => {
        setActiveMode(mode);
        setForm((current) => ({ ...current, mode, sortBy: "hot" }));
        setDashboardVisible(false);
    };

    const currentMode = modeByValue[activeMode];

    return (
        <div
            style={{
                minHeight: "100vh",
                background: "#0d0806",
                color: "white",
                fontFamily: '"DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                position: "relative",
                overflowX: "hidden",
            }}
        >
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        html{scroll-behavior:smooth}
        body{margin:0}
        ::placeholder{color:rgba(255,255,255,0.24)}
        ::-webkit-scrollbar{width:6px;height:6px}
        ::-webkit-scrollbar-thumb{background:rgba(255,69,0,0.36);border-radius:999px}
        .ambient{position:fixed;inset:0;pointer-events:none;overflow:hidden;z-index:0}
        .orb{position:absolute;border-radius:999px;filter:blur(1px);opacity:0.9;pointer-events:none}
        .orb1{top:-180px;left:-160px;width:560px;height:560px;background:radial-gradient(circle,rgba(255,69,0,0.26) 0%,transparent 70%);animation:drift1 19s ease-in-out infinite}
        .orb2{top:-80px;right:-120px;width:420px;height:420px;background:radial-gradient(circle,rgba(255,123,77,0.17) 0%,transparent 70%);animation:drift2 23s ease-in-out infinite}
        .orb3{bottom:-120px;left:18%;width:360px;height:360px;background:radial-gradient(circle,rgba(255,69,0,0.13) 0%,transparent 70%);animation:drift3 15s ease-in-out infinite}
        .orb4{bottom:-110px;right:18%;width:320px;height:320px;background:radial-gradient(circle,rgba(255,69,0,0.11) 0%,transparent 70%);animation:drift4 21s ease-in-out infinite}
        @keyframes drift1{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(55px,-45px) scale(1.07)}66%{transform:translate(-28px,52px) scale(0.94)}}
        @keyframes drift2{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(-65px,48px) scale(1.06)}66%{transform:translate(42px,-55px) scale(1.09)}}
        @keyframes drift3{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(28px,44px) scale(0.9)}}
        @keyframes drift4{0%,100%{transform:translate(0,0)}40%{transform:translate(-48px,-28px)}70%{transform:translate(22px,58px)}}
        .heroFade{animation:heroFade 0.7s ease both}
        .lift{animation:lift 0.7s cubic-bezier(.22,1,.36,1) both}
        .lift2{animation:lift 0.7s cubic-bezier(.22,1,.36,1) 0.08s both}
        .lift3{animation:lift 0.7s cubic-bezier(.22,1,.36,1) 0.16s both}
        .lift4{animation:lift 0.7s cubic-bezier(.22,1,.36,1) 0.24s both}
        @keyframes heroFade{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
        @keyframes lift{from{opacity:0;transform:translateY(14px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
      `}</style>

            <div className="ambient">
                <div className="orb orb1" />
                <div className="orb orb2" />
                <div className="orb orb3" />
                <div className="orb orb4" />
            </div>

            <header
                style={{
                    position: "sticky",
                    top: 0,
                    zIndex: 20,
                    backdropFilter: "blur(24px)",
                    background: "rgba(13,8,6,0.58)",
                    borderBottom: "1px solid rgba(255,255,255,0.06)",
                }}
            >
                <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                        <RedditLogo size={32} />
                        <div>
                            <div style={{ fontSize: 16, fontWeight: 800, letterSpacing: -0.5 }}>Reddit Lens</div>
                            <div style={{ fontSize: 11.5, color: "rgba(255,255,255,0.34)" }}>Live sentiment intelligence</div>
                        </div>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: 10, color: "rgba(255,255,255,0.36)", fontSize: 12, flexWrap: "wrap", justifyContent: "flex-end" }}>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                            <span style={{ width: 8, height: 8, borderRadius: 999, background: "#6EE7B7", boxShadow: "0 0 10px #6EE7B7" }} />
                            Backend ready: {API_BASE.replace(/^https?:\/\//, "")}
                        </span>
                    </div>
                </div>
            </header>

            <main style={{ maxWidth: 1200, margin: "0 auto", padding: "0 28px 70px", position: "relative", zIndex: 1 }}>
                <section style={{ textAlign: "center", padding: "56px 0 28px" }} className="heroFade">
                    <div style={{ display: "inline-flex", alignItems: "center", gap: 10, padding: "6px 14px", borderRadius: 999, background: "rgba(255,69,0,0.10)", border: "1px solid rgba(255,69,0,0.22)" }}>
                        <span style={{ fontSize: 11.5, fontWeight: 800, letterSpacing: 1.1, color: REDDIT_ORANGE_LIGHT, textTransform: "uppercase" }}>Sentiment dashboard</span>
                    </div>

                    <h1 style={{ margin: "18px 0 12px", fontSize: 56, lineHeight: 1.02, letterSpacing: -2.6, fontWeight: 900, background: "linear-gradient(180deg, #fff 0%, rgba(255,255,255,0.74) 45%, rgba(255,180,120,0.42) 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                        What is Reddit actually feeling?
                    </h1>

                    <p style={{ maxWidth: 640, margin: "0 auto", color: "rgba(255,255,255,0.36)", fontSize: 16, lineHeight: 1.75 }}>
                        Start with a subreddit, a post URL, multiple communities, or any custom text. The dashboard only opens after the analysis returns from Python.
                    </p>
                </section>

                <section className="lift" style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 12, marginBottom: 16 }}>
                    {modeOptions.map((mode) => {
                        const active = activeMode === mode.value;
                        return (
                            <button
                                key={mode.value}
                                type="button"
                                onClick={() => handleModeChange(mode.value)}
                                style={{
                                    textAlign: "left",
                                    borderRadius: 24,
                                    border: `1px solid ${active ? "rgba(255,69,0,0.45)" : "rgba(255,255,255,0.08)"}`,
                                    background: active ? "rgba(255,69,0,0.09)" : "rgba(255,255,255,0.04)",
                                    color: "white",
                                    padding: 18,
                                    cursor: "pointer",
                                    boxShadow: active ? "0 18px 38px rgba(255,69,0,0.14)" : "0 14px 32px rgba(0,0,0,0.18)",
                                    backdropFilter: "blur(20px)",
                                    transition: "all 0.2s ease",
                                    minHeight: 132,
                                }}
                            >
                                <div style={{ fontSize: 12, letterSpacing: 1.1, color: active ? REDDIT_ORANGE_LIGHT : "rgba(255,255,255,0.30)", fontWeight: 800, textTransform: "uppercase", marginBottom: 10 }}>
                                    {String(mode.value).replace(/_/g, " ")}
                                </div>
                                <div style={{ fontSize: 16, fontWeight: 800, lineHeight: 1.3, marginBottom: 8 }}>{mode.label}</div>
                                <div style={{ fontSize: 12.5, color: "rgba(255,255,255,0.30)", lineHeight: 1.6 }}>{mode.description}</div>
                            </button>
                        );
                    })}
                </section>

                <section className="lift2" style={{ marginBottom: 18 }}>
                    <GlassPanel style={{ padding: 18 }}>
                        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                            <div>
                                <div style={{ fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: "rgba(255,255,255,0.32)", marginBottom: 10, fontWeight: 800 }}>Input</div>
                                {activeMode === "subreddit" && (
                                    <input
                                        value={form.subreddit}
                                        onChange={(event) => updateField("subreddit", event.target.value)}
                                        placeholder={currentMode?.placeholder}
                                        style={inputStyle}
                                    />
                                )}
                                {activeMode === "post_comments" && (
                                    <input
                                        value={form.postUrl}
                                        onChange={(event) => updateField("postUrl", event.target.value)}
                                        placeholder={currentMode?.placeholder}
                                        style={inputStyle}
                                    />
                                )}
                                {activeMode === "compare" && (
                                    <input
                                        value={form.compareText}
                                        onChange={(event) => updateField("compareText", event.target.value)}
                                        placeholder={currentMode?.placeholder}
                                        style={inputStyle}
                                    />
                                )}
                                {activeMode === "custom_text" && (
                                    <textarea
                                        value={form.customText}
                                        onChange={(event) => updateField("customText", event.target.value)}
                                        placeholder={currentMode?.placeholder}
                                        style={{ ...inputStyle, minHeight: 118, resize: "vertical", paddingTop: 14 }}
                                    />
                                )}
                            </div>

                            <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
                                {activeMode !== "custom_text" && (
                                    <label style={fieldLabelStyle}>
                                        <span style={fieldLabelText}>Limit</span>
                                        <input
                                            value={activeMode === "post_comments" ? form.commentLimit : form.limit}
                                            onChange={(event) => updateField(activeMode === "post_comments" ? "commentLimit" : "limit", event.target.value)}
                                            type="number"
                                            min="1"
                                            max="100"
                                            style={smallInputStyle}
                                        />
                                    </label>
                                )}

                                {activeMode !== "custom_text" && (
                                    <div style={{ display: "flex", gap: 8 }}>
                                        {["hot", "new", "top"].map((sort) => (
                                            <button
                                                key={sort}
                                                onClick={() => updateField("sortBy", sort)}
                                                style={{
                                                    padding: "12px 16px",
                                                    borderRadius: 18,
                                                    border: `1px solid ${form.sortBy === sort ? REDDIT_ORANGE + "60" : "rgba(255,255,255,0.1)"}`,
                                                    background: form.sortBy === sort ? REDDIT_ORANGE + "18" : "rgba(255,255,255,0.04)",
                                                    backdropFilter: "blur(12px)",
                                                    color: form.sortBy === sort ? REDDIT_ORANGE : "rgba(255,255,255,0.6)",
                                                    cursor: "pointer",
                                                    transition: "all 0.22s ease",
                                                    fontSize: 13,
                                                    fontWeight: form.sortBy === sort ? 700 : 600,
                                                    textTransform: "capitalize",
                                                    outline: "none",
                                                    fontFamily: "inherit",
                                                    minWidth: 60,
                                                }}
                                            >
                                                {sort}
                                            </button>
                                        ))}
                                    </div>
                                )}

                                <button
                                    type="button"
                                    onClick={handleAnalyze}
                                    disabled={loading}
                                    style={{
                                        minWidth: 148,
                                        borderRadius: 18,
                                        border: "none",
                                        padding: "12px 20px",
                                        fontSize: 14,
                                        fontWeight: 800,
                                        color: "white",
                                        background: loading ? "linear-gradient(135deg, rgba(255,69,0,0.55), rgba(255,123,77,0.55))" : `linear-gradient(135deg, ${REDDIT_ORANGE}, ${REDDIT_ORANGE_LIGHT})`,
                                        boxShadow: "0 14px 30px rgba(255,69,0,0.24)",
                                        cursor: loading ? "wait" : "pointer",
                                        marginLeft: "auto",
                                    }}
                                >
                                    {loading ? "Analyzing..." : "Analyze"}
                                </button>
                            </div>
                        </div>

                    </GlassPanel>
                </section>

                {dashboardVisible ? (
                    <section ref={dashboardRef} className="lift3" style={{ marginTop: 24 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, marginBottom: 18, flexWrap: "wrap" }}>
                            <div>
                                <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: 1.1, color: "rgba(255,255,255,0.28)", fontWeight: 800 }}>Dashboard</div>
                                <h2 style={{ margin: "10px 0 0", fontSize: 28, letterSpacing: -1, fontWeight: 900 }}>{result?.title || "Analysis results"}</h2>
                                <div style={{ marginTop: 8, color: "rgba(255,255,255,0.34)", fontSize: 13.5, lineHeight: 1.6 }}>{result?.insight || error || "The dashboard updates with the backend response as soon as you analyze."}</div>
                            </div>

                            {result?.request ? (
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "flex-end" }}>
                                    <span style={requestChipStyle}>Mode: {result.request.mode || requestMode}</span>
                                    {result.request.sort_by ? <span style={requestChipStyle}>Sort: {result.request.sort_by}</span> : null}
                                    {result.request.subreddit ? <span style={requestChipStyle}>r/{result.request.subreddit}</span> : null}
                                    {Array.isArray(result.request.subreddits) ? <span style={requestChipStyle}>{result.request.subreddits.length} subreddits</span> : null}
                                </div>
                            ) : null}
                        </div>

                        {error && !result ? (
                            <GlassPanel style={{ padding: 22, marginBottom: 16, borderColor: "rgba(252,165,165,0.24)" }}>
                                <div style={{ fontSize: 13, fontWeight: 800, color: sentimentPalette.Negative.color, marginBottom: 8 }}>Backend error</div>
                                <div style={{ color: "rgba(255,255,255,0.68)", lineHeight: 1.7 }}>{error}</div>
                            </GlassPanel>
                        ) : null}

                        {result ? (
                            <>
                                <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: 12, marginBottom: 14 }}>
                                    <MetricCard label="Total items" value={formatCount(summary.total)} hint="Posts, comments, or text blocks" accent={REDDIT_ORANGE_LIGHT} />
                                    <MetricCard label="Positive" value={formatPercent(summary.positive_pct ?? summary.positive_percent ?? 0)} hint="Share of positive results" accent={sentimentPalette.Positive.color} />
                                    <MetricCard label="Neutral" value={formatPercent(summary.neutral_pct ?? 0)} hint="Confidence band results" accent={sentimentPalette.Neutral.color} />
                                    <MetricCard label="Negative" value={formatPercent(summary.negative_pct ?? 0)} hint="Share of negative results" accent={sentimentPalette.Negative.color} />
                                    <MetricCard label="Average score" value={formatScore(summary.avg_score ?? 0)} hint="Model output average" accent="#FFD8C2" />
                                </div>

                                <div style={{ display: "grid", gridTemplateColumns: "0.82fr 1.18fr", gap: 14, marginBottom: 14 }}>
                                    <GlassPanel style={{ padding: 22 }}>
                                        <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1.1, color: "rgba(255,255,255,0.28)", fontWeight: 800, marginBottom: 12 }}>Sentiment mix</div>
                                        <div style={{ display: "flex", gap: 18, alignItems: "center", flexWrap: "wrap" }}>
                                            <Donut summary={summary} />
                                            <div style={{ display: "grid", gap: 10, minWidth: 210 }}>
                                                {[
                                                    ["Positive", summary.positive, summary.positive_pct],
                                                    ["Neutral", summary.neutral, summary.neutral_pct],
                                                    ["Negative", summary.negative, summary.negative_pct],
                                                ].map(([label, count, pct]) => {
                                                    const palette = sentimentPalette[label];
                                                    return (
                                                        <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
                                                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                                <span style={{ width: 8, height: 8, borderRadius: 999, background: palette.color, boxShadow: `0 0 10px ${palette.color}66` }} />
                                                                <span style={{ color: "rgba(255,255,255,0.38)", fontSize: 12.5 }}>{label}</span>
                                                            </div>
                                                            <div style={{ color: palette.color, fontSize: 13, fontWeight: 800 }}>{formatCount(count)} · {formatPercent(pct)}</div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    </GlassPanel>

                                    <GlassPanel style={{ padding: 22 }}>
                                        <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1.1, color: "rgba(255,255,255,0.28)", fontWeight: 800, marginBottom: 14 }}>Top keywords</div>
                                        <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                                            {keywordItems.length ? keywordItems.map((keyword, index) => {
                                                const term = typeof keyword === "string" ? keyword : keyword.term || keyword.word || keyword.label;
                                                const count = typeof keyword === "object" ? keyword.count ?? keyword.score ?? null : null;
                                                return (
                                                    <div
                                                        key={`${term}-${index}`}
                                                        style={{
                                                            padding: "7px 12px",
                                                            borderRadius: 999,
                                                            background: "rgba(255,255,255,0.05)",
                                                            border: "1px solid rgba(255,255,255,0.08)",
                                                            color: "rgba(255,255,255,0.76)",
                                                            fontSize: 12.5,
                                                            fontWeight: 700,
                                                        }}
                                                    >
                                                        {term}{count !== null ? <span style={{ color: REDDIT_ORANGE_LIGHT, marginLeft: 8 }}>{count}</span> : null}
                                                    </div>
                                                );
                                            }) : <div style={{ color: "rgba(255,255,255,0.28)", fontSize: 13 }}>The backend can return keywords for each analysis.</div>}
                                        </div>
                                    </GlassPanel>
                                </div>

                                {comparisonItems.length ? (
                                    <div style={{ marginBottom: 14 }}>
                                        <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1.1, color: "rgba(255,255,255,0.28)", fontWeight: 800, margin: "0 0 12px" }}>Compare results</div>
                                        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12 }}>
                                            {comparisonItems.map((item) => <ComparisonCard key={item.subreddit} item={item} />)}
                                        </div>
                                    </div>
                                ) : null}

                                <GlassPanel style={{ padding: 20 }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
                                        <div>
                                            <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1.1, color: "rgba(255,255,255,0.28)", fontWeight: 800 }}>Results</div>
                                            <div style={{ marginTop: 6, color: "rgba(255,255,255,0.40)", fontSize: 13.5 }}>{sortedItems.length} entries shown from the backend</div>
                                        </div>

                                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                                            {["all", "Positive", "Neutral", "Negative"].map((sentiment) => (
                                                <button
                                                    key={sentiment}
                                                    type="button"
                                                    onClick={() => setSentimentFilter(sentiment)}
                                                    style={{
                                                        borderRadius: 999,
                                                        padding: "8px 14px",
                                                        border: `1px solid ${sentimentFilter === sentiment ? "rgba(255,69,0,0.36)" : "rgba(255,255,255,0.08)"}`,
                                                        background: sentimentFilter === sentiment ? "rgba(255,69,0,0.12)" : "rgba(255,255,255,0.04)",
                                                        color: sentimentFilter === sentiment ? "white" : "rgba(255,255,255,0.34)",
                                                        fontSize: 12.5,
                                                        fontWeight: 700,
                                                        cursor: "pointer",
                                                    }}
                                                >
                                                    {sentiment}
                                                </button>
                                            ))}

                                            <div style={{ width: 1, height: 24, background: "rgba(255,255,255,0.08)" }} />

                                            {[
                                                { value: "score", label: "Score" },
                                                { value: "upvotes", label: "Upvotes" },
                                                { value: "activity", label: "Activity" },
                                                { value: "recent", label: "Recent" },
                                            ].map((option) => (
                                                <button
                                                    key={option.value}
                                                    type="button"
                                                    onClick={() => setListSort(option.value)}
                                                    style={{
                                                        borderRadius: 999,
                                                        padding: "8px 14px",
                                                        border: `1px solid ${listSort === option.value ? "rgba(255,69,0,0.36)" : "rgba(255,255,255,0.08)"}`,
                                                        background: listSort === option.value ? "rgba(255,69,0,0.12)" : "rgba(255,255,255,0.04)",
                                                        color: listSort === option.value ? "white" : "rgba(255,255,255,0.34)",
                                                        fontSize: 12.5,
                                                        fontWeight: 700,
                                                        cursor: "pointer",
                                                    }}
                                                >
                                                    {option.label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    <div style={{ display: "grid", gap: 12 }}>
                                        {sortedItems.length ? sortedItems.map((item, index) => <ResultCard key={item.id || `${item.title || item.text || index}`} item={item} />) : <div style={{ color: "rgba(255,255,255,0.32)", fontSize: 13.5 }}>No results to show.</div>}
                                    </div>
                                </GlassPanel>
                            </>
                        ) : null}
                    </section>
                ) : null}
            </main>
        </div>
    );
}

const inputStyle = {
    width: "100%",
    borderRadius: 18,
    border: "1px solid rgba(255,255,255,0.08)",
    background: "rgba(255,255,255,0.04)",
    color: "white",
    padding: "14px 16px",
    fontSize: 14,
    fontWeight: 600,
    outline: "none",
};

const smallInputStyle = {
    width: "100%",
    borderRadius: 18,
    border: "1px solid rgba(255,255,255,0.08)",
    background: "rgba(255,255,255,0.04)",
    color: "white",
    padding: "14px 14px",
    fontSize: 14,
    fontWeight: 700,
    outline: "none",
};

const fieldLabelStyle = {
    display: "flex",
    flexDirection: "column",
    gap: 10,
    minWidth: 0,
};

const fieldLabelText = {
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 1,
    color: "rgba(255,255,255,0.32)",
    fontWeight: 800,
};

const requestChipStyle = {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    padding: "7px 12px",
    borderRadius: 999,
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    color: "rgba(255,255,255,0.32)",
    fontSize: 12,
    fontWeight: 700,
};
