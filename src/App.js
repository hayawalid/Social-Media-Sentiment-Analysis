import { useState, useEffect } from "react";

const R = "#FF4500";
const R2 = "#FF6534";

/* ── icons ─────────────────────────────────────────── */
const Ico = ({ children, size = 15, color = "currentColor" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
    style={{ flexShrink: 0, display:"block" }}>
    {children}
  </svg>
);

const IcoUp     = ({color,size}) => <Ico color={color} size={size}><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></Ico>;
const IcoMsg    = ({color,size}) => <Ico color={color} size={size}><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></Ico>;
const IcoCal    = ({color,size}) => <Ico color={color} size={size}><rect x="3" y="4" width="18" height="18" rx="3"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></Ico>;
const IcoFilter = ({color,size}) => <Ico color={color} size={size}><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></Ico>;
const IcoExport = ({color,size}) => <Ico color={color} size={size}><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></Ico>;
const IcoHappy  = ({color,size}) => <Ico color={color} size={size}><circle cx="12" cy="12" r="9"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></Ico>;
const IcoMeh    = ({color,size}) => <Ico color={color} size={size}><circle cx="12" cy="12" r="9"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></Ico>;
const IcoSad    = ({color,size}) => <Ico color={color} size={size}><circle cx="12" cy="12" r="9"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></Ico>;

const RedditLogo = ({ size = 30 }) => (
  <svg width={size} height={size} viewBox="0 0 20 20">
    <circle cx="10" cy="10" r="10" fill={R}/>
    <path d="M16.67 10a1.46 1.46 0 00-2.47-1 7.12 7.12 0 00-3.85-1.23l.65-3.07 2.12.45a1 1 0 101.07-1 1 1 0 00-.95.68l-2.38-.5a.15.15 0 00-.18.11l-.73 3.44a7.14 7.14 0 00-3.89 1.23 1.46 1.46 0 10-1.61 2.39 2.89 2.89 0 000 .44c0 2.24 2.61 4.06 5.83 4.06s5.83-1.82 5.83-4.06a2.89 2.89 0 000-.44 1.46 1.46 0 00.49-1.5zm-9.6 1.33a1 1 0 111 1 1 1 0 01-1-1zm5.56 2.64a3.47 3.47 0 01-2.63.82 3.47 3.47 0 01-2.63-.82.2.2 0 01.28-.28 3.09 3.09 0 002.35.66 3.09 3.09 0 002.35-.66.2.2 0 01.28.28zm-.2-1.64a1 1 0 111-1 1 1 0 01-1 1z" fill="white" stroke="none"/>
  </svg>
);

/* ── data ───────────────────────────────────────────── */
const comments = [
  { id:1, user:"velvet_cosmos", sub:"r/technology", text:"Completely changed how I think about this. One of those rare moments where you feel genuinely smarter after reading.", sentiment:"positive", score:0.93, ups:12400, replies:287, daysAgo:2 },
  { id:2, user:"neon_static",   sub:"r/programming",text:"Mediocre at best. Expected depth, got surface-level takes and a comment section full of echo chambers.", sentiment:"negative", score:-0.72, ups:3820,  replies:94,  daysAgo:5 },
  { id:3, user:"morningfog_",   sub:"r/science",    text:"Not groundbreaking but solid. A few genuinely interesting points buried in a lot of filler.", sentiment:"neutral",  score:0.08, ups:1940,  replies:42,  daysAgo:1 },
  { id:4, user:"quartz_mind",   sub:"r/technology", text:"The research depth here is unreal. Whoever wrote this spent serious time with the source material.", sentiment:"positive", score:0.89, ups:8760,  replies:156, daysAgo:7 },
  { id:5, user:"gravel_wave",   sub:"r/gadgets",    text:"I've seen more original takes from a fortune cookie. Rehashed, poorly sourced, forgettable.", sentiment:"negative", score:-0.85, ups:5130,  replies:412, daysAgo:3 },
  { id:6, user:"soft_atlas",    sub:"r/science",    text:"Works for what it is. The visuals help a lot and the structure is clean even if nothing surprises you.", sentiment:"neutral",  score:0.14, ups:2210,  replies:31,  daysAgo:9 },
  { id:7, user:"driftwood99",   sub:"r/gadgets",    text:"I keep coming back to this. Every re-read surfaces something new. This is the standard.", sentiment:"positive", score:0.97, ups:21300, replies:543, daysAgo:14 },
  { id:8, user:"pale_junction", sub:"r/technology", text:"45 minutes I won't get back. Repetitive, bloated, and the conclusion barely follows from the evidence.", sentiment:"negative", score:-0.81, ups:890,   replies:178, daysAgo:4 },
];

const scfg = {
  positive: { color:"#6EE7B7", glow:"rgba(110,231,183,0.18)", label:"Positive",  Ico: IcoHappy },
  neutral:  { color:"#C4C4D4", glow:"rgba(196,196,212,0.12)", label:"Neutral",   Ico: IcoMeh   },
  negative: { color:"#FCA5A5", glow:"rgba(252,165,165,0.18)", label:"Negative",  Ico: IcoSad   },
};

const fmt = n => n >= 1000 ? (n/1000).toFixed(1)+"K" : String(n);

/* ── animated background ───────────────────────────── */
function Background() {
  return (
    <div style={{ position:"fixed", inset:0, overflow:"hidden", zIndex:0, pointerEvents:"none" }}>
      <div className="orb orb1" />
      <div className="orb orb2" />
      <div className="orb orb3" />
      <div className="orb orb4" />
    </div>
  );
}

/* ── glass card ─────────────────────────────────────── */
function Glass({ children, style={} }) {
  const [hov, setHov] = useState(false);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? "rgba(255,255,255,0.075)" : "rgba(255,255,255,0.048)",
        backdropFilter: "blur(32px) saturate(180%)",
        WebkitBackdropFilter: "blur(32px) saturate(180%)",
        border: `1px solid rgba(255,255,255,${hov?0.14:0.09})`,
        borderRadius: 24,
        boxShadow: hov
          ? "0 28px 64px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.16)"
          : "0 12px 36px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.08)",
        transform: hov ? "translateY(-4px) scale(1.005)" : "translateY(0) scale(1)",
        transition: "all 0.38s cubic-bezier(0.34,1.3,0.64,1)",
        position: "relative",
        overflow: "hidden",
        ...style,
      }}
    >{children}</div>
  );
}

/* ── pill button ────────────────────────────────────── */
function Pill({ children, active, onClick, accent }) {
  const c = accent || R;
  return (
    <button onClick={onClick} style={{
      padding: "7px 16px", fontSize: 12.5, fontWeight: 600, borderRadius: 99,
      border: `1px solid ${active ? c+"60" : "rgba(255,255,255,0.1)"}`,
      background: active ? c+"18" : "rgba(255,255,255,0.04)",
      backdropFilter: "blur(12px)",
      color: active ? c : "rgba(255,255,255,0.38)",
      cursor: "pointer", transition: "all 0.22s ease",
    }}>{children}</button>
  );
}

/* ── donut chart ────────────────────────────────────── */
function Donut({ pos, neu, neg }) {
  const total = pos + neu + neg;
  const r = 46, C = 2 * Math.PI * r, gap = 3;
  const slices = [
    { v: pos, color: "#6EE7B7" },
    { v: neu, color: "#C4C4D4" },
    { v: neg, color: "#FCA5A5" },
  ];
  let offset = C * 0.25;
  return (
    <svg width={120} height={120} viewBox="0 0 120 120">
      <circle cx={60} cy={60} r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={12}/>
      {slices.map(({ v, color }, i) => {
        const sl = (v / total) * C - gap;
        const el = (
          <circle key={i} cx={60} cy={60} r={r} fill="none" stroke={color} strokeWidth={12}
            strokeDasharray={`${Math.max(0, sl)} ${C}`}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 7px ${color}90)` }}
          />
        );
        offset -= (v / total) * C;
        return el;
      })}
      <text x={60} y={54} textAnchor="middle" fill="white" fontSize={18} fontWeight={800} fontFamily="DM Sans,sans-serif">
        {Math.round((pos / total) * 100)}%
      </text>
      <text x={60} y={70} textAnchor="middle" fill="rgba(255,255,255,0.32)" fontSize={9.5} fontFamily="DM Sans,sans-serif">positive</text>
    </svg>
  );
}

/* ── score bar ──────────────────────────────────────── */
function ScoreBar({ score }) {
  const pct = ((score + 1) / 2) * 100;
  const color = score > 0.2 ? "#6EE7B7" : score < -0.2 ? "#FCA5A5" : "#C4C4D4";
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10 }}>
      <div style={{ flex:1, height:3, background:"rgba(255,255,255,0.07)", borderRadius:4 }}>
        <div style={{ width:`${pct}%`, height:"100%", background:`linear-gradient(90deg,${color}70,${color})`, borderRadius:4, boxShadow:`0 0 8px ${color}55`, transition:"width 1s cubic-bezier(0.34,1.2,0.64,1)" }}/>
      </div>
      <span style={{ fontSize:11, fontWeight:700, color, fontFamily:"monospace", minWidth:38, textAlign:"right" }}>
        {score > 0 ? "+" : ""}{score.toFixed(2)}
      </span>
    </div>
  );
}

/* ── comment card ───────────────────────────────────── */
function CommentCard({ c, idx, sortKey }) {
  const s = scfg[c.sentiment];
  const [vis, setVis] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVis(true), idx * 60 + 80);
    return () => clearTimeout(t);
  }, []);

  return (
    <div style={{
      opacity: vis ? 1 : 0,
      transform: vis ? "translateY(0) scale(1)" : "translateY(20px) scale(0.97)",
      transition: `opacity 0.5s ease ${idx*0.05}s, transform 0.55s cubic-bezier(0.34,1.3,0.64,1) ${idx*0.05}s`,
    }}>
      <Glass style={{ padding:"20px 22px" }}>
        {/* inner glow */}
        <div style={{ position:"absolute", top:-24, right:-16, width:110, height:110, borderRadius:"50%", background:`radial-gradient(circle,${s.glow},transparent 70%)`, pointerEvents:"none" }}/>

        <div style={{ display:"flex", gap:13, alignItems:"flex-start", position:"relative" }}>
          {/* avatar bubble */}
          <div style={{ width:38, height:38, borderRadius:"50%", background:"radial-gradient(135deg,rgba(255,69,0,0.25),rgba(255,100,52,0.08))", border:"1px solid rgba(255,255,255,0.11)", display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
            <span style={{ fontSize:11, fontWeight:800, color:"rgba(255,255,255,0.65)" }}>{c.user.slice(0,2).toUpperCase()}</span>
          </div>

          <div style={{ flex:1, minWidth:0 }}>
            {/* header row */}
            <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", gap:8, marginBottom:8 }}>
              <div style={{ display:"flex", alignItems:"center", gap:7, flexWrap:"wrap" }}>
                <span style={{ fontSize:13, fontWeight:700, color:"rgba(255,255,255,0.82)" }}>u/{c.user}</span>
                <span style={{ fontSize:11, color:"rgba(255,255,255,0.25)", background:"rgba(255,255,255,0.05)", padding:"2px 9px", borderRadius:99 }}>{c.sub}</span>
                <span style={{ fontSize:10, color:"rgba(255,255,255,0.18)" }}>{c.daysAgo}d ago</span>
              </div>
              {/* sentiment bubble */}
              <div style={{ display:"flex", alignItems:"center", gap:5, background:s.glow, border:`1px solid ${s.color}35`, borderRadius:99, padding:"4px 11px", flexShrink:0, boxShadow:`0 0 14px ${s.color}25` }}>
                <s.Ico color={s.color} size={13}/>
                <span style={{ fontSize:11, color:s.color, fontWeight:600, letterSpacing:0.2 }}>{s.label}</span>
              </div>
            </div>

            <p style={{ fontSize:13.5, color:"rgba(255,255,255,0.5)", lineHeight:1.72, margin:"0 0 14px", fontFamily:"DM Sans,sans-serif", fontWeight:400 }}>{c.text}</p>
            <ScoreBar score={c.score}/>

            {/* meta */}
            <div style={{ display:"flex", gap:20, marginTop:11 }}>
              {[
                { Comp:IcoUp,  val:fmt(c.ups),     active: sortKey==="ups" },
                { Comp:IcoMsg, val:`${fmt(c.replies)} replies`, active: sortKey==="replies" },
                { Comp:IcoCal, val:`${c.daysAgo}d`,active: sortKey==="date" },
              ].map(({ Comp, val, active }, i) => (
                <div key={i} style={{ display:"flex", alignItems:"center", gap:5 }}>
                  <Comp color={active ? R2 : "rgba(255,255,255,0.2)"} size={13}/>
                  <span style={{ fontSize:12, color: active ? "rgba(255,255,255,0.6)" : "rgba(255,255,255,0.22)", fontWeight: active ? 600 : 400 }}>{val}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Glass>
    </div>
  );
}

/* ── app ────────────────────────────────────────────── */
export default function App() {
  const [tab, setTab]       = useState(0);
  const [query, setQuery]   = useState("");
  const [sort, setSort]     = useState("ups");
  const [sortDir, setSortDir] = useState("desc");
  const [filter, setFilter] = useState("all");
  const [focused, setFocused] = useState(false);

  const pos = comments.filter(c => c.sentiment === "positive").length;
  const neu = comments.filter(c => c.sentiment === "neutral").length;
  const neg = comments.filter(c => c.sentiment === "negative").length;
  const total = comments.length;

  const sorted = [...comments]
    .filter(c => filter === "all" || c.sentiment === filter)
    .sort((a, b) => {
      const av = sort === "date" ? -a.daysAgo : (sort === "replies" ? a.replies : a.ups);
      const bv = sort === "date" ? -b.daysAgo : (sort === "replies" ? b.replies : b.ups);
      return sortDir === "desc" ? bv - av : av - bv;
    });

  const toggleSort = k => {
    if (sort === k) setSortDir(d => d === "desc" ? "asc" : "desc");
    else { setSort(k); setSortDir("desc"); }
  };

  const tabs = ["Subreddit", "Post Comments", "Compare", "Custom Text"];
  const placeholders = ["r/subreddit name…", "Paste post URL…", "r/tech vs r/science", "Paste your text…"];

  return (
    <div style={{ minHeight:"100vh", background:"#0d0806", fontFamily:"DM Sans,-apple-system,sans-serif", color:"white", position:"relative", overflowX:"hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::placeholder{color:rgba(255,255,255,0.22);font-family:DM Sans,sans-serif}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-thumb{background:rgba(255,69,0,0.35);border-radius:4px}
        @keyframes drift1{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(55px,-45px) scale(1.07)}66%{transform:translate(-28px,52px) scale(0.94)}}
        @keyframes drift2{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(-65px,48px) scale(1.06)}66%{transform:translate(42px,-55px) scale(1.09)}}
        @keyframes drift3{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(28px,44px) scale(0.9)}}
        @keyframes drift4{0%,100%{transform:translate(0,0)}40%{transform:translate(-48px,-28px)}70%{transform:translate(22px,58px)}}
        @keyframes floatUp{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}
        @keyframes floatB{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
        @keyframes floatC{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
        @keyframes fadeSlide{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}
        @keyframes livePulse{0%,100%{opacity:1;box-shadow:0 0 6px #6EE7B7}50%{opacity:0.45;box-shadow:0 0 2px #6EE7B7}}
        .orb{position:absolute;border-radius:50%;pointer-events:none}
        .orb1{top:-180px;left:-140px;width:560px;height:560px;background:radial-gradient(circle,rgba(255,69,0,0.26) 0%,transparent 70%);animation:drift1 19s ease-in-out infinite;filter:blur(1px)}
        .orb2{top:-90px;right:-90px;width:400px;height:400px;background:radial-gradient(circle,rgba(255,140,0,0.17) 0%,transparent 70%);animation:drift2 23s ease-in-out infinite}
        .orb3{top:42%;left:52%;width:280px;height:280px;background:radial-gradient(circle,rgba(255,100,52,0.13) 0%,transparent 70%);animation:drift3 15s ease-in-out infinite}
        .orb4{bottom:-110px;right:18%;width:320px;height:320px;background:radial-gradient(circle,rgba(255,69,0,0.11) 0%,transparent 70%);animation:drift4 21s ease-in-out infinite}
        .sc1{animation:floatUp 6.2s ease-in-out infinite}
        .sc2{animation:floatB 7.4s ease-in-out infinite 0.6s}
        .sc3{animation:floatC 5.8s ease-in-out infinite 1.1s}
        .sc4{animation:floatUp 8.1s ease-in-out infinite 1.7s}
        .h1{animation:fadeSlide 0.75s cubic-bezier(0.34,1.2,0.64,1) both}
        .h2{animation:fadeSlide 0.75s cubic-bezier(0.34,1.2,0.64,1) 0.1s both}
        .h3{animation:fadeSlide 0.75s cubic-bezier(0.34,1.2,0.64,1) 0.18s both}
        .h4{animation:fadeSlide 0.75s cubic-bezier(0.34,1.2,0.64,1) 0.26s both}
        .live{animation:livePulse 2.2s ease-in-out infinite;width:7px;height:7px;border-radius:50%;background:#6EE7B7}
      `}</style>

      <Background/>

      {/* nav */}
      <nav style={{ position:"sticky", top:0, zIndex:50, height:58, padding:"0 40px", display:"flex", alignItems:"center", gap:16, background:"rgba(13,8,6,0.55)", backdropFilter:"blur(28px)", borderBottom:"1px solid rgba(255,255,255,0.055)" }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <RedditLogo size={30}/>
          <span style={{ fontSize:16, fontWeight:800, letterSpacing:-0.6 }}>Reddit Lens</span>
        </div>
        <div style={{ flex:1 }}/>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div className="live"/>
          <span style={{ fontSize:12, color:"rgba(255,255,255,0.3)", fontWeight:500 }}>r/technology · live</span>
        </div>
      </nav>

      <div style={{ maxWidth:1060, margin:"0 auto", padding:"0 32px 80px", position:"relative", zIndex:1 }}>

        {/* hero */}
        <div style={{ textAlign:"center", padding:"60px 0 44px" }}>
          <div className="h1">
            <div style={{ display:"inline-flex", alignItems:"center", gap:8, background:"rgba(255,69,0,0.1)", border:"1px solid rgba(255,69,0,0.22)", borderRadius:99, padding:"5px 16px", marginBottom:22, backdropFilter:"blur(12px)" }}>
              <RedditLogo size={14}/>
              <span style={{ fontSize:11.5, color:"rgba(255,120,60,0.9)", fontWeight:700, letterSpacing:0.9 }}>SENTIMENT ANALYSIS</span>
            </div>
            <h1 style={{ fontSize:54, fontWeight:800, letterSpacing:-2.5, lineHeight:1.06, marginBottom:14, background:"linear-gradient(155deg,#fff 45%,rgba(255,180,120,0.65))", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>
              What is Reddit<br/>actually feeling?
            </h1>
          </div>
          <p className="h2" style={{ fontSize:15.5, color:"rgba(255,255,255,0.33)", maxWidth:420, margin:"0 auto 34px", lineHeight:1.72 }}>
            Paste a subreddit, post URL, or any text — get sentiment decoded instantly.
          </p>

          {/* tabs */}
          <div className="h3" style={{ display:"inline-flex", gap:6, flexWrap:"wrap", justifyContent:"center", marginBottom:26 }}>
            {tabs.map((t, i) => <Pill key={i} active={tab === i} onClick={() => setTab(i)}>{t}</Pill>)}
          </div>

          {/* input */}
          <div className="h4" style={{ maxWidth:540, margin:"0 auto" }}>
            <div style={{
              display:"flex", alignItems:"center",
              background: focused ? "rgba(255,255,255,0.09)" : "rgba(255,255,255,0.05)",
              border: `1px solid ${focused ? "rgba(255,69,0,0.45)" : "rgba(255,255,255,0.1)"}`,
              borderRadius:99, overflow:"hidden", backdropFilter:"blur(24px)",
              boxShadow: focused ? "0 0 0 4px rgba(255,69,0,0.09), 0 14px 44px rgba(0,0,0,0.32)" : "0 8px 32px rgba(0,0,0,0.26)",
              transition:"all 0.3s ease",
            }}>
              <div style={{ padding:"0 14px 0 18px", opacity:0.28, display:"flex" }}>
                <IcoFilter color="white" size={14}/>
              </div>
              <input
                value={query}
                onChange={e => setQuery(e.target.value)}
                onFocus={() => setFocused(true)}
                onBlur={() => setFocused(false)}
                placeholder={placeholders[tab]}
                style={{ flex:1, background:"transparent", border:"none", outline:"none", color:"white", fontSize:14, padding:"14px 0", fontFamily:"DM Sans,sans-serif", fontWeight:500 }}
              />
              <button style={{ margin:5, padding:"10px 22px", background:`linear-gradient(135deg,${R},${R2})`, border:"none", borderRadius:99, color:"white", fontSize:13, fontWeight:700, cursor:"pointer", boxShadow:"0 4px 22px rgba(255,69,0,0.42)", letterSpacing:0.2, whiteSpace:"nowrap" }}>
                Analyze
              </button>
            </div>
          </div>
        </div>

        {/* stat bubbles */}
        <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:13, marginBottom:13 }}>
          {[
            { label:"Positive",  val:`${Math.round((pos/total)*100)}%`, color:"#6EE7B7", note:"of comments", cls:"sc1" },
            { label:"Avg Score", val:"+0.31",   color:R2,       note:"leaning positive", cls:"sc2" },
            { label:"Analyzed",  val:"10.8K",   color:"rgba(255,255,255,0.72)", note:"total comments", cls:"sc3" },
            { label:"Negative",  val:`${Math.round((neg/total)*100)}%`, color:"#FCA5A5", note:"of comments", cls:"sc4" },
          ].map(({ label, val, color, note, cls }) => (
            <Glass key={label} style={{ padding:"22px 20px" }}>
              <div className={cls} style={{ position:"absolute", top:-18, right:-18, width:76, height:76, borderRadius:"50%", background:`radial-gradient(circle,${color}20,transparent 70%)`, pointerEvents:"none" }}/>
              <div style={{ fontSize:10.5, color:"rgba(255,255,255,0.3)", letterSpacing:1.1, marginBottom:10, fontWeight:700 }}>{label.toUpperCase()}</div>
              <div style={{ fontSize:31, fontWeight:800, color, letterSpacing:-1.4, lineHeight:1 }}>{val}</div>
              <div style={{ fontSize:11, color:"rgba(255,255,255,0.22)", marginTop:7 }}>{note}</div>
            </Glass>
          ))}
        </div>

        {/* bento row */}
        <div style={{ display:"grid", gridTemplateColumns:"196px 1fr 1fr", gap:13, marginBottom:13 }}>

          {/* donut card */}
          <Glass style={{ padding:"24px 18px", display:"flex", flexDirection:"column", alignItems:"center", gap:14 }}>
            <div style={{ position:"absolute", inset:0, background:"radial-gradient(ellipse at 50% 110%,rgba(255,69,0,0.09),transparent 70%)", pointerEvents:"none", borderRadius:24 }}/>
            <span style={{ fontSize:10.5, color:"rgba(255,255,255,0.28)", letterSpacing:1.1, fontWeight:700, alignSelf:"flex-start" }}>SENTIMENT</span>
            <Donut pos={pos} neu={neu} neg={neg}/>
            <div style={{ display:"flex", flexDirection:"column", gap:8, width:"100%" }}>
              {[["#6EE7B7","Positive",pos],["#C4C4D4","Neutral",neu],["#FCA5A5","Negative",neg]].map(([color,label,n])=>(
                <div key={label} style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                  <div style={{ display:"flex", alignItems:"center", gap:7 }}>
                    <div style={{ width:7, height:7, borderRadius:"50%", background:color, boxShadow:`0 0 7px ${color}` }}/>
                    <span style={{ fontSize:11.5, color:"rgba(255,255,255,0.38)" }}>{label}</span>
                  </div>
                  <span style={{ fontSize:12, fontWeight:700, color }}>{Math.round((n/total)*100)}%</span>
                </div>
              ))}
            </div>
          </Glass>

          {/* keywords */}
          <Glass style={{ padding:"24px 22px" }}>
            <span style={{ fontSize:10.5, color:"rgba(255,255,255,0.28)", letterSpacing:1.1, fontWeight:700, display:"block", marginBottom:16 }}>TOP KEYWORDS</span>
            <div style={{ display:"flex", flexWrap:"wrap", gap:7 }}>
              {[["brilliant",0.91],["mediocre",-0.6],["depth",0.7],["repetitive",-0.75],["research",0.6],["bloated",-0.7],["innovative",0.85],["surface-level",-0.5],["rare",0.6],["forgettable",-0.55],["solid",0.3],["original",0.5]].map(([word, score]) => {
                const color = score > 0.2 ? "#6EE7B7" : score < -0.2 ? "#FCA5A5" : "#C4C4D4";
                return (
                  <div key={word} style={{ padding:"5px 13px", borderRadius:99, background:`${color}15`, border:`1px solid ${color}28`, fontSize:12.5, color, fontWeight:500, backdropFilter:"blur(8px)" }}>
                    {word}
                  </div>
                );
              })}
            </div>
          </Glass>

          {/* distribution */}
          <Glass style={{ padding:"24px 22px" }}>
            <span style={{ fontSize:10.5, color:"rgba(255,255,255,0.28)", letterSpacing:1.1, fontWeight:700, display:"block", marginBottom:16 }}>DISTRIBUTION</span>
            <div style={{ display:"flex", flexDirection:"column", gap:11 }}>
              {[["Very Positive",28,"#6EE7B7"],["Positive",22,"#86EFAC"],["Neutral",25,"#C4C4D4"],["Negative",15,"#F87171"],["Very Negative",10,"#FCA5A5"]].map(([label,pct,color])=>(
                <div key={label}>
                  <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
                    <span style={{ fontSize:12, color:"rgba(255,255,255,0.33)" }}>{label}</span>
                    <span style={{ fontSize:12, color, fontWeight:700 }}>{pct}%</span>
                  </div>
                  <div style={{ height:4, background:"rgba(255,255,255,0.05)", borderRadius:99 }}>
                    <div style={{ width:`${pct*3.2}%`, height:"100%", background:`linear-gradient(90deg,${color}55,${color})`, borderRadius:99, boxShadow:`0 0 8px ${color}40` }}/>
                  </div>
                </div>
              ))}
            </div>
          </Glass>
        </div>

        {/* comments */}
        <div style={{ marginTop:34 }}>
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:12, marginBottom:18 }}>
            <div style={{ display:"flex", alignItems:"baseline", gap:10 }}>
              <span style={{ fontSize:20, fontWeight:800, letterSpacing:-0.5 }}>Comments</span>
              <span style={{ fontSize:12.5, color:"rgba(255,255,255,0.22)" }}>{sorted.length} shown</span>
            </div>

            <div style={{ display:"flex", gap:7, alignItems:"center", flexWrap:"wrap" }}>
              {/* sentiment filter */}
              {[["all","all",R],["positive","Positive","#6EE7B7"],["neutral","Neutral","#C4C4D4"],["negative","Negative","#FCA5A5"]].map(([val,label,color])=>(
                <Pill key={val} active={filter===val} onClick={()=>setFilter(val)} accent={color}>{label}</Pill>
              ))}

              <div style={{ width:1, height:18, background:"rgba(255,255,255,0.1)", margin:"0 1px" }}/>

              {/* sort */}
              {[["ups","Likes",IcoUp],["replies","Replies",IcoMsg],["date","Date",IcoCal]].map(([key,label,Comp])=>(
                <button key={key} onClick={()=>toggleSort(key)} style={{
                  display:"flex", alignItems:"center", gap:6, padding:"7px 14px", borderRadius:99,
                  border:`1px solid ${sort===key?"rgba(255,69,0,0.4)":"rgba(255,255,255,0.09)"}`,
                  background: sort===key ? "rgba(255,69,0,0.12)" : "rgba(255,255,255,0.035)",
                  backdropFilter:"blur(12px)",
                  color: sort===key ? "rgba(255,255,255,0.82)" : "rgba(255,255,255,0.32)",
                  fontSize:12.5, fontWeight: sort===key ? 600 : 400, cursor:"pointer", transition:"all 0.2s",
                }}>
                  <Comp color={sort===key ? R2 : "rgba(255,255,255,0.28)"} size={13}/>
                  {label}
                  {sort===key && <span style={{ fontSize:10, opacity:0.55 }}>{sortDir==="desc"?"↓":"↑"}</span>}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
            {sorted.map((c, i) => <CommentCard key={c.id} c={c} idx={i} sortKey={sort}/>)}
          </div>
        </div>

        {/* footer */}
        <div style={{ marginTop:50, paddingTop:22, borderTop:"1px solid rgba(255,255,255,0.06)", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <div style={{ display:"flex", alignItems:"center", gap:9 }}>
            <RedditLogo size={19}/>
            <span style={{ fontSize:12, color:"rgba(255,255,255,0.2)" }}>Reddit Lens · NLP Sentiment Engine</span>
          </div>
          <div style={{ display:"flex", gap:6 }}>
            {["Export","Share","Schedule"].map(l => (
              <button key={l} style={{ display:"flex", alignItems:"center", gap:6, padding:"6px 14px", fontSize:12, border:"1px solid rgba(255,255,255,0.08)", borderRadius:99, background:"rgba(255,255,255,0.03)", backdropFilter:"blur(8px)", color:"rgba(255,255,255,0.3)", cursor:"pointer" }}>
                {l === "Export" && <IcoExport color="rgba(255,255,255,0.3)" size={13}/>}
                {l}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}