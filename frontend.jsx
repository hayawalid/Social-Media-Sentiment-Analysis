import { useState, useEffect, useRef } from "react";

const REDDIT_ORANGE = "#FF4500";
const REDDIT_ORANGE_DARK = "#CC3700";
const REDDIT_ORANGE_GLOW = "rgba(255,69,0,0.15)";
const REDDIT_ORANGE_DIM = "rgba(255,69,0,0.08)";

const RedditSVG = ({ size = 32 }) => (
  <svg width={size} height={size} viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="10" cy="10" r="10" fill={REDDIT_ORANGE} />
    <path d="M16.67 10a1.46 1.46 0 00-2.47-1 7.12 7.12 0 00-3.85-1.23l.65-3.07 2.12.45a1 1 0 101.07-1 1 1 0 00-.95.68l-2.38-.5a.15.15 0 00-.18.11l-.73 3.44a7.14 7.14 0 00-3.89 1.23 1.46 1.46 0 10-1.61 2.39 2.89 2.89 0 000 .44c0 2.24 2.61 4.06 5.83 4.06s5.83-1.82 5.83-4.06a2.89 2.89 0 000-.44 1.46 1.46 0 00.49-1.5zm-9.6 1.33a1 1 0 111 1 1 1 0 01-1-1zm5.56 2.64a3.47 3.47 0 01-2.63.82 3.47 3.47 0 01-2.63-.82.2.2 0 01.28-.28 3.09 3.09 0 002.35.66 3.09 3.09 0 002.35-.66.2.2 0 01.28.28zm-.2-1.64a1 1 0 111-1 1 1 0 01-1 1z" fill="white"/>
  </svg>
);

const mockComments = [
  { id: 1, author: "u/techEnthusiast", text: "This is absolutely incredible! Best update they've ever released, completely changed my workflow.", sentiment: "positive", score: 0.92, upvotes: 1243, sub: "r/technology" },
  { id: 2, author: "u/daily_coder", text: "Pretty mediocre honestly. I expected way more from this release. Very disappointing overall.", sentiment: "negative", score: -0.74, upvotes: 87, sub: "r/programming" },
  { id: 3, author: "u/neutralObserver", text: "It has some good parts and some bad parts. Overall it's just okay, nothing groundbreaking.", sentiment: "neutral", score: 0.05, upvotes: 341, sub: "r/technology" },
  { id: 4, author: "u/PowerUser99", text: "Genuinely love this. The performance improvements alone make it worth it. Absolutely recommended.", sentiment: "positive", score: 0.88, upvotes: 892, sub: "r/gadgets" },
  { id: 5, author: "u/skeptic_sam", text: "Total garbage. Broken features everywhere, bugs galore, and zero documentation. Avoid.", sentiment: "negative", score: -0.91, upvotes: 213, sub: "r/technology" },
  { id: 6, author: "u/midrange_mike", text: "Works fine for my use case. Not revolutionary but gets the job done without major issues.", sentiment: "neutral", score: 0.12, upvotes: 156, sub: "r/programming" },
  { id: 7, author: "u/fanboy_frank", text: "Mind-blowing innovation. This sets the standard for everything going forward. Absolute perfection.", sentiment: "positive", score: 0.95, upvotes: 2041, sub: "r/gadgets" },
  { id: 8, author: "u/realist_raj", text: "The hype is overblown. Sure it works, but nothing here you haven't seen before. Marketing magic.", sentiment: "negative", score: -0.48, upvotes: 504, sub: "r/technology" },
];

const sentimentColors = {
  positive: { bg: "rgba(52, 199, 89, 0.12)", border: "rgba(52, 199, 89, 0.3)", text: "#34C759", label: "Positive" },
  neutral: { bg: "rgba(255, 214, 10, 0.10)", border: "rgba(255, 214, 10, 0.3)", text: "#FFD60A", label: "Neutral" },
  negative: { bg: "rgba(255, 69, 58, 0.10)", border: "rgba(255, 69, 58, 0.3)", text: "#FF453A", label: "Negative" },
};

const SentimentBar = ({ value }) => {
  const pct = ((value + 1) / 2) * 100;
  const color = value > 0.2 ? "#34C759" : value < -0.2 ? "#FF453A" : "#FFD60A";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 4, background: "rgba(255,255,255,0.08)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2, transition: "width 1s cubic-bezier(0.34,1.56,0.64,1)" }} />
      </div>
      <span style={{ fontSize: 11, color: color, fontWeight: 600, minWidth: 36, textAlign: "right", fontFamily: "SF Mono, monospace" }}>
        {value > 0 ? "+" : ""}{value.toFixed(2)}
      </span>
    </div>
  );
};

const DonutChart = ({ positive, neutral, negative }) => {
  const total = positive + neutral + negative;
  const r = 52;
  const cx = 70, cy = 70;
  const circumference = 2 * Math.PI * r;
  const posSlice = (positive / total) * circumference;
  const neuSlice = (neutral / total) * circumference;
  const negSlice = (negative / total) * circumference;
  const gap = 2;

  return (
    <svg width={140} height={140} viewBox="0 0 140 140">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={14} />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#34C759" strokeWidth={14}
        strokeDasharray={`${posSlice - gap} ${circumference - posSlice + gap}`}
        strokeDashoffset={circumference * 0.25} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#FFD60A" strokeWidth={14}
        strokeDasharray={`${neuSlice - gap} ${circumference - neuSlice + gap}`}
        strokeDashoffset={circumference * 0.25 - posSlice + gap} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#FF453A" strokeWidth={14}
        strokeDasharray={`${negSlice - gap} ${circumference - negSlice + gap}`}
        strokeDashoffset={circumference * 0.25 - posSlice - neuSlice + gap * 2} strokeLinecap="round" />
      <text x={cx} y={cy - 6} textAnchor="middle" fill="white" fontSize={20} fontWeight={700} fontFamily="-apple-system, BlinkMacSystemFont, SF Pro Display">
        {Math.round((positive / total) * 100)}%
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" fill="rgba(255,255,255,0.45)" fontSize={10} fontFamily="-apple-system, BlinkMacSystemFont, SF Pro Text">
        positive
      </text>
    </svg>
  );
};

const BarChart = ({ data }) => {
  const max = Math.max(...data.map(d => d.value));
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 6, height: 80, padding: "0 4px" }}>
      {data.map((d, i) => (
        <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
          <div style={{ width: "100%", height: Math.max(4, (d.value / max) * 72), background: d.color, borderRadius: "4px 4px 0 0", transition: `height 0.8s cubic-bezier(0.34,1.56,0.64,1) ${i * 0.05}s` }} />
          <span style={{ fontSize: 9, color: "rgba(255,255,255,0.35)", textAlign: "center", lineHeight: 1 }}>{d.label}</span>
        </div>
      ))}
    </div>
  );
};

const CommentCard = ({ comment, index }) => {
  const s = sentimentColors[comment.sentiment];
  const [visible, setVisible] = useState(false);
  useEffect(() => { setTimeout(() => setVisible(true), index * 60); }, []);
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      border: `0.5px solid rgba(255,255,255,0.07)`,
      borderRadius: 16,
      padding: "14px 16px",
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(12px)",
      transition: `opacity 0.5s ease ${index * 0.06}s, transform 0.5s cubic-bezier(0.34,1.56,0.64,1) ${index * 0.06}s`,
      cursor: "default",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: "50%", background: `linear-gradient(135deg, ${REDDIT_ORANGE_GLOW}, rgba(255,69,0,0.3))`, border: `1px solid rgba(255,69,0,0.3)`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={{ fontSize: 10, color: REDDIT_ORANGE, fontWeight: 700 }}>{comment.author.slice(2, 4).toUpperCase()}</span>
          </div>
          <div>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", fontWeight: 500 }}>{comment.author}</div>
            <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)" }}>{comment.sub}</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ background: s.bg, border: `0.5px solid ${s.border}`, color: s.text, fontSize: 10, fontWeight: 600, padding: "3px 8px", borderRadius: 6 }}>{s.label}</span>
        </div>
      </div>
      <p style={{ fontSize: 12.5, color: "rgba(255,255,255,0.6)", lineHeight: 1.6, margin: "0 0 10px", fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text" }}>{comment.text}</p>
      <SentimentBar value={comment.score} />
      <div style={{ marginTop: 8, fontSize: 10, color: "rgba(255,255,255,0.25)", display: "flex", gap: 12 }}>
        <span>▲ {comment.upvotes.toLocaleString()} upvotes</span>
      </div>
    </div>
  );
};

const BentoCard = ({ children, style = {}, className = "" }) => (
  <div style={{
    background: "rgba(255,255,255,0.035)",
    border: "0.5px solid rgba(255,255,255,0.08)",
    borderRadius: 24,
    padding: 20,
    backdropFilter: "blur(20px)",
    ...style
  }}>
    {children}
  </div>
);

const StatPill = ({ label, value, color }) => (
  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)" }}>
    <span style={{ fontSize: 12, color: "rgba(255,255,255,0.4)" }}>{label}</span>
    <span style={{ fontSize: 13, fontWeight: 600, color }}>{value}</span>
  </div>
);

const tabs = ["Subreddit", "Post Comments", "Compare", "Custom Text"];
const menuItems = [
  { icon: "🔮", label: "Subreddit Analysis", desc: "Analyze overall sentiment of any subreddit" },
  { icon: "💬", label: "Post Comments", desc: "Deep-dive into a specific post's comments" },
  { icon: "⚖️", label: "Compare Subreddits", desc: "Side-by-side sentiment comparison" },
  { icon: "✏️", label: "Custom Text", desc: "Paste any text for instant analysis" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [query, setQuery] = useState("");
  const [showDashboard, setShowDashboard] = useState(true);
  const [hoveredComment, setHoveredComment] = useState(null);
  const [filterSentiment, setFilterSentiment] = useState("all");

  const positive = mockComments.filter(c => c.sentiment === "positive").length;
  const neutral = mockComments.filter(c => c.sentiment === "neutral").length;
  const negative = mockComments.filter(c => c.sentiment === "negative").length;
  const total = mockComments.length;
  const avgScore = (mockComments.reduce((a, c) => a + c.score, 0) / total).toFixed(2);

  const filteredComments = filterSentiment === "all" ? mockComments : mockComments.filter(c => c.sentiment === filterSentiment);

  const weekData = [
    { label: "Mon", value: 72, color: "#34C759" },
    { label: "Tue", value: 45, color: "#FFD60A" },
    { label: "Wed", value: 88, color: "#34C759" },
    { label: "Thu", value: 31, color: "#FF453A" },
    { label: "Fri", value: 65, color: "#34C759" },
    { label: "Sat", value: 57, color: "#FFD60A" },
    { label: "Sun", value: 79, color: "#34C759" },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0A0A0A",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', sans-serif",
      color: "white",
      overflowX: "hidden",
    }}>
      {/* Ambient background */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
        <div style={{ position: "absolute", top: -200, left: "50%", transform: "translateX(-50%)", width: 700, height: 700, background: `radial-gradient(ellipse at center, ${REDDIT_ORANGE_GLOW} 0%, transparent 70%)`, filter: "blur(40px)" }} />
        <div style={{ position: "absolute", top: 0, right: 0, width: 300, height: 300, background: "radial-gradient(ellipse at top right, rgba(255,69,0,0.06) 0%, transparent 70%)" }} />
        <div style={{ position: "absolute", bottom: 0, left: 0, width: 400, height: 300, background: "radial-gradient(ellipse at bottom left, rgba(255,69,0,0.04) 0%, transparent 70%)" }} />
      </div>

      <div style={{ position: "relative", zIndex: 1, maxWidth: 1100, margin: "0 auto", padding: "0 24px 60px" }}>

        {/* Header */}
        <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "28px 0 0" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <RedditSVG size={36} />
            <div>
              <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: -0.4, color: "white" }}>Reddit Lens</div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", letterSpacing: 0.2 }}>Sentiment Intelligence</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#34C759", boxShadow: "0 0 8px #34C759" }} />
            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>Live · r/technology</span>
          </div>
        </header>

        {/* Hero input section */}
        <section style={{ textAlign: "center", padding: "52px 0 40px" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: REDDIT_ORANGE_DIM, border: `0.5px solid rgba(255,69,0,0.2)`, borderRadius: 20, padding: "5px 14px", marginBottom: 20 }}>
            <span style={{ fontSize: 11, color: REDDIT_ORANGE, fontWeight: 600, letterSpacing: 0.8 }}>REDDIT SENTIMENT ANALYSIS TOOL</span>
          </div>
          <h1 style={{ fontSize: 48, fontWeight: 800, letterSpacing: -1.5, lineHeight: 1.1, margin: "0 0 12px", background: "linear-gradient(180deg, #fff 60%, rgba(255,255,255,0.4))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Decode the Crowd.
          </h1>
          <p style={{ fontSize: 16, color: "rgba(255,255,255,0.4)", maxWidth: 480, margin: "0 auto 36px", lineHeight: 1.6 }}>
            Understand how Reddit truly feels — in real time.
          </p>

          {/* Mode tabs */}
          <div style={{ display: "inline-flex", gap: 2, background: "rgba(255,255,255,0.04)", borderRadius: 14, padding: 4, marginBottom: 28 }}>
            {tabs.map((t, i) => (
              <button key={i} onClick={() => setActiveTab(i)} style={{
                padding: "8px 16px", fontSize: 12.5, fontWeight: 500, borderRadius: 10, border: "none", cursor: "pointer",
                background: activeTab === i ? "rgba(255,255,255,0.1)" : "transparent",
                color: activeTab === i ? "white" : "rgba(255,255,255,0.35)",
                transition: "all 0.2s ease",
              }}>{t}</button>
            ))}
          </div>

          {/* Input bar */}
          <div style={{ maxWidth: 560, margin: "0 auto", position: "relative" }}>
            <div style={{ display: "flex", alignItems: "center", background: "rgba(255,255,255,0.06)", border: `1px solid rgba(255,255,255,0.1)`, borderRadius: 16, overflow: "hidden", transition: "border 0.2s" }}>
              <span style={{ padding: "0 14px", fontSize: 15, color: "rgba(255,255,255,0.25)" }}>
                {activeTab === 0 ? "r/" : activeTab === 1 ? "🔗" : activeTab === 2 ? "⚖️" : "✏️"}
              </span>
              <input value={query} onChange={e => setQuery(e.target.value)}
                placeholder={["Enter subreddit name...", "Paste post URL...", "e.g. r/tech vs r/science", "Paste your text here..."][activeTab]}
                style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "white", fontSize: 14, padding: "14px 0", fontFamily: "inherit" }}
              />
              <button onClick={() => setShowDashboard(true)} style={{
                margin: 5, padding: "9px 20px", background: REDDIT_ORANGE, border: "none", borderRadius: 11, color: "white", fontSize: 13, fontWeight: 600, cursor: "pointer",
                transition: "all 0.2s ease", boxShadow: `0 4px 20px rgba(255,69,0,0.35)`,
              }}>Analyze →</button>
            </div>
          </div>
        </section>

        {/* Quick select options */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 44 }}>
          {menuItems.map((item, i) => (
            <div key={i} onClick={() => setActiveTab(i)} style={{
              background: activeTab === i ? REDDIT_ORANGE_DIM : "rgba(255,255,255,0.025)",
              border: `0.5px solid ${activeTab === i ? "rgba(255,69,0,0.25)" : "rgba(255,255,255,0.06)"}`,
              borderRadius: 16, padding: "14px 16px", cursor: "pointer",
              transition: "all 0.25s ease",
            }}>
              <div style={{ fontSize: 20, marginBottom: 6 }}>{item.icon}</div>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: activeTab === i ? REDDIT_ORANGE : "rgba(255,255,255,0.75)", marginBottom: 3 }}>{item.label}</div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", lineHeight: 1.4 }}>{item.desc}</div>
            </div>
          ))}
        </div>

        {/* Dashboard */}
        {showDashboard && (
          <div>
            {/* Section label */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
              <div style={{ height: 1, flex: 1, background: "rgba(255,255,255,0.06)" }} />
              <span style={{ fontSize: 11, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, fontWeight: 600 }}>ANALYSIS DASHBOARD · r/technology</span>
              <div style={{ height: 1, flex: 1, background: "rgba(255,255,255,0.06)" }} />
            </div>

            {/* Magic Bento Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gridTemplateRows: "auto auto", gap: 14, marginBottom: 14 }}>

              {/* Donut chart card - tall */}
              <BentoCard style={{ gridRow: "1 / 3", display: "flex", flexDirection: "column", gap: 20 }}>
                <div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", letterSpacing: 1, marginBottom: 4 }}>OVERALL SENTIMENT</div>
                  <div style={{ fontSize: 17, fontWeight: 700, color: "white" }}>r/technology</div>
                </div>
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <DonutChart positive={positive} neutral={neutral} negative={negative} />
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  {[{ label: "Positive", color: "#34C759", pct: Math.round((positive / total) * 100) },
                    { label: "Neutral", color: "#FFD60A", pct: Math.round((neutral / total) * 100) },
                    { label: "Negative", color: "#FF453A", pct: Math.round((negative / total) * 100) }].map(({ label, color, pct }) => (
                    <div key={label} style={{ flex: 1, textAlign: "center" }}>
                      <div style={{ fontSize: 18, fontWeight: 700, color }}>{pct}%</div>
                      <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", marginTop: 2 }}>{label}</div>
                    </div>
                  ))}
                </div>
                <div style={{ borderTop: "0.5px solid rgba(255,255,255,0.07)", paddingTop: 14 }}>
                  <StatPill label="Total Comments" value={`${total}`} color="rgba(255,255,255,0.7)" />
                  <StatPill label="Avg. Score" value={avgScore > 0 ? `+${avgScore}` : avgScore} color={avgScore > 0 ? "#34C759" : "#FF453A"} />
                  <StatPill label="Top Upvoted" value="2,041 ▲" color={REDDIT_ORANGE} />
                  <StatPill label="Confidence" value="94.3%" color="rgba(255,255,255,0.7)" />
                </div>
              </BentoCard>

              {/* Score card */}
              <BentoCard style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", letterSpacing: 1 }}>SENTIMENT SCORE</div>
                <div>
                  <div style={{ fontSize: 52, fontWeight: 800, color: "#34C759", letterSpacing: -2, lineHeight: 1 }}>+0.31</div>
                  <div style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", marginTop: 4 }}>Leaning Positive</div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 11, color: "#34C759" }}>▲ 12.4%</span>
                  <span style={{ fontSize: 11, color: "rgba(255,255,255,0.25)" }}>vs. last week</span>
                </div>
              </BentoCard>

              {/* Momentum card */}
              <BentoCard style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", letterSpacing: 1 }}>WEEKLY MOMENTUM</div>
                <BarChart data={weekData} />
              </BentoCard>

              {/* Activity card */}
              <BentoCard style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", letterSpacing: 1, marginBottom: 4 }}>COMMENT VOLUME</div>
                <div style={{ display: "flex", gap: 12 }}>
                  {[{ label: "Posts", val: "1.2k" }, { label: "Comments", val: "48k" }, { label: "Unique Users", val: "9.4k" }].map(({ label, val }) => (
                    <div key={label} style={{ flex: 1 }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color: "white" }}>{val}</div>
                      <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)" }}>{label}</div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 4, height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 2 }}>
                  <div style={{ width: "68%", height: "100%", background: `linear-gradient(90deg, ${REDDIT_ORANGE}, rgba(255,69,0,0.4))`, borderRadius: 2 }} />
                </div>
                <div style={{ fontSize: 10, color: "rgba(255,255,255,0.25)" }}>68% of daily average</div>
              </BentoCard>
            </div>

            {/* Top keywords bento */}
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14, marginBottom: 14 }}>
              <BentoCard>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", letterSpacing: 1, marginBottom: 14 }}>TOP KEYWORDS</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {[
                    { word: "performance", score: 0.82, count: 143 },
                    { word: "broken", score: -0.79, count: 89 },
                    { word: "incredible", score: 0.91, count: 67 },
                    { word: "mediocre", score: -0.51, count: 54 },
                    { word: "innovative", score: 0.88, count: 112 },
                    { word: "disappointing", score: -0.74, count: 76 },
                    { word: "recommend", score: 0.75, count: 94 },
                    { word: "garbage", score: -0.95, count: 38 },
                    { word: "love", score: 0.89, count: 201 },
                    { word: "okay", score: 0.05, count: 145 },
                    { word: "perfect", score: 0.96, count: 88 },
                    { word: "bugs", score: -0.82, count: 62 },
                  ].map(({ word, score, count }) => {
                    const c = score > 0.2 ? { bg: "rgba(52,199,89,0.12)", border: "rgba(52,199,89,0.25)", text: "#34C759" }
                      : score < -0.2 ? { bg: "rgba(255,69,58,0.10)", border: "rgba(255,69,58,0.25)", text: "#FF453A" }
                      : { bg: "rgba(255,214,10,0.10)", border: "rgba(255,214,10,0.25)", text: "#FFD60A" };
                    return (
                      <div key={word} style={{ background: c.bg, border: `0.5px solid ${c.border}`, borderRadius: 8, padding: "5px 10px", display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ fontSize: 12.5, color: c.text, fontWeight: 500 }}>{word}</span>
                        <span style={{ fontSize: 10, color: "rgba(255,255,255,0.25)" }}>×{count}</span>
                      </div>
                    );
                  })}
                </div>
              </BentoCard>

              <BentoCard style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", letterSpacing: 1 }}>DISTRIBUTION</div>
                {[
                  { label: "Very Positive", pct: 28, color: "#34C759" },
                  { label: "Positive", pct: 19, color: "#5AC87C" },
                  { label: "Neutral", pct: 25, color: "#FFD60A" },
                  { label: "Negative", pct: 16, color: "#FF6B61" },
                  { label: "Very Negative", pct: 12, color: "#FF453A" },
                ].map(({ label, pct, color }) => (
                  <div key={label}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>{label}</span>
                      <span style={{ fontSize: 11, color, fontWeight: 600 }}>{pct}%</span>
                    </div>
                    <div style={{ height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 2 }}>
                      <div style={{ width: `${pct * 3}%`, height: "100%", background: color, borderRadius: 2 }} />
                    </div>
                  </div>
                ))}
              </BentoCard>
            </div>

            {/* Comments section */}
            <div style={{ marginTop: 28 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "white" }}>Comment Analysis</div>
                  <div style={{ fontSize: 12, color: "rgba(255,255,255,0.3)" }}>{filteredComments.length} comments analyzed</div>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  {["all", "positive", "neutral", "negative"].map(f => (
                    <button key={f} onClick={() => setFilterSentiment(f)} style={{
                      padding: "6px 14px", fontSize: 11.5, fontWeight: 500, borderRadius: 8, border: `0.5px solid ${filterSentiment === f ? "rgba(255,69,0,0.4)" : "rgba(255,255,255,0.1)"}`,
                      background: filterSentiment === f ? REDDIT_ORANGE_DIM : "rgba(255,255,255,0.03)",
                      color: filterSentiment === f ? REDDIT_ORANGE : "rgba(255,255,255,0.4)",
                      cursor: "pointer", transition: "all 0.2s", textTransform: "capitalize",
                    }}>{f}</button>
                  ))}
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {filteredComments.map((comment, i) => (
                  <CommentCard key={comment.id} comment={comment} index={i} />
                ))}
              </div>
            </div>

            {/* Footer strip */}
            <div style={{ marginTop: 40, padding: "20px 0", borderTop: "0.5px solid rgba(255,255,255,0.05)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <RedditSVG size={20} />
                <span style={{ fontSize: 12, color: "rgba(255,255,255,0.25)" }}>Reddit Lens · Powered by NLP</span>
              </div>
              <div style={{ display: "flex", gap: 4 }}>
                {["Export", "Share", "Schedule"].map(label => (
                  <button key={label} style={{
                    padding: "6px 12px", fontSize: 11, border: "0.5px solid rgba(255,255,255,0.1)", borderRadius: 8,
                    background: "transparent", color: "rgba(255,255,255,0.35)", cursor: "pointer",
                  }}>{label}</button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}