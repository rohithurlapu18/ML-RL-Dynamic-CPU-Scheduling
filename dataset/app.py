
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Consistent dark chart palette used by the dashboard.
PLOTLY_DARK = "plotly_dark"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Dynamic CPU Scheduling | Review 2",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DARK PROFESSIONAL THEME
# ============================================================
st.markdown("""
<style>
/* ---------- Global ---------- */
html, body, [class*="css"] {
    color: #E8F0F8 !important;
}

.stApp {
    background: #07111F !important;
    color: #E8F0F8 !important;
}

header[data-testid="stHeader"] {
    background: #07111F !important;
}

footer {
    background: #07111F !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #091827 !important;
    border-right: 1px solid #203B58 !important;
}

[data-testid="stSidebar"] * {
    color: #D9E6F3 !important;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #D9E6F3 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #D9E6F3 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: #FFFFFF !important;
}

/* ---------- Headings / text ---------- */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
}

.stMarkdown p,
.stMarkdown li,
.stMarkdown {
    color: #C1D0DF !important;
}

/* ---------- Hero ---------- */
.hero {
    background:
        radial-gradient(circle at 85% 20%, rgba(67, 148, 255, .15), transparent 35%),
        linear-gradient(135deg, #0D2036 0%, #102A47 100%);
    border: 1px solid #244767;
    border-radius: 24px;
    padding: 32px 36px;
    margin-bottom: 22px;
    box-shadow: 0 16px 50px rgba(0,0,0,.28);
}

.hero h1 {
    margin: 0;
    color: #FFFFFF !important;
    font-size: 40px;
    letter-spacing: -.6px;
}

.hero p {
    color: #AFC3D8 !important;
    font-size: 16px;
    margin-top: 9px;
}

/* ---------- Sections ---------- */
.section {
    background: #0C1B2D !important;
    border: 1px solid #1D3854;
    border-radius: 19px;
    padding: 24px;
    margin: 15px 0;
    box-shadow: 0 10px 34px rgba(0,0,0,.22);
}

.section h1,
.section h2,
.section h3,
.section h4 {
    color: #FFFFFF !important;
}

/* ---------- Metrics ---------- */
.metric {
    background: linear-gradient(145deg, #10243A, #0C1C2E) !important;
    border: 1px solid #244766;
    border-radius: 17px;
    padding: 19px;
    box-shadow: 0 8px 25px rgba(0,0,0,.18);
}

.metric-label {
    color: #8FA8C0 !important;
    font-size: 12px;
}

.metric-value {
    color: #FFFFFF !important;
    font-size: 28px;
    font-weight: 800;
    margin-top: 3px;
}

.metric-note {
    color: #8096AD !important;
    font-size: 11px;
}

/* ---------- Process cards ---------- */
.step {
    background: #0E2135 !important;
    border: 1px solid #23445F;
    border-radius: 17px;
    padding: 19px;
    min-height: 155px;
    box-shadow: 0 8px 25px rgba(0,0,0,.17);
}

.step-num {
    display: inline-block;
    background: #12385D !important;
    color: #6FB5FF !important;
    border: 1px solid #2B6090;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    text-align: center;
    line-height: 36px;
    font-weight: 800;
    margin-bottom: 9px;
}

.step-title {
    font-weight: 750;
    color: #FFFFFF !important;
    font-size: 16px;
}

.step-text {
    color: #AFC1D4 !important;
    font-size: 13px;
    line-height: 1.6;
}

/* ---------- Tags ---------- */
.tag {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    background: #102E4D !important;
    color: #72B7FF !important;
    border: 1px solid #24547D;
    font-size: 11px;
    margin: 3px;
}

.good {
    background: #103529 !important;
    color: #6FE1B0 !important;
    border-color: #236B51 !important;
}

.warn {
    background: #3A2B16 !important;
    color: #FFD078 !important;
    border-color: #765322 !important;
}

.info {
    background: #102E4D !important;
    color: #72B7FF !important;
}

/* ---------- Streamlit widgets ---------- */
.stButton button {
    background: #123250 !important;
    color: #FFFFFF !important;
    border: 1px solid #2A5B86 !important;
}

.stButton button:hover {
    background: #174369 !important;
    border-color: #4D9AE8 !important;
}

[data-baseweb="select"],
[data-baseweb="select"] > div,
[data-baseweb="input"],
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
    background: #0E2135 !important;
    color: #E8F0F8 !important;
    border-color: #2A4965 !important;
}

[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-baseweb="textarea"] * {
    color: #E8F0F8 !important;
}

[data-baseweb="popover"] {
    background: #0E2135 !important;
    border: 1px solid #2A4965 !important;
}

[data-baseweb="popover"] * {
    color: #E8F0F8 !important;
}

.stSlider label,
.stSelectbox label,
.stMultiSelect label,
.stTextInput label,
.stNumberInput label {
    color: #C8D6E4 !important;
}

/* ---------- Expanders / alerts ---------- */
[data-testid="stExpander"] {
    background: #0C1B2D !important;
    border: 1px solid #1D3854 !important;
}

[data-testid="stExpander"] * {
    color: #D8E4EF !important;
}

[data-testid="stAlert"] {
    background: #0D2135 !important;
    border-color: #2A4E6D !important;
}

[data-testid="stAlert"] * {
    color: #DCE8F4 !important;
}

/* ---------- Dataframes ---------- */
[data-testid="stDataFrame"] {
    background: #0C1B2D !important;
    border: 1px solid #1D3854 !important;
}

[data-testid="stDataFrame"] * {
    color: #DCE8F4 !important;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #617B95 !important;
    padding: 30px 0 10px;
    font-size: 11px;
}

/* ---------- Sidebar collapse / toolbar icons ---------- */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stToolbar"] button {
    color: #BBD0E5 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).resolve().parent
DATASET = BASE / "task_machine_candidates_v2.csv"
PPO_RESULTS = BASE / "ppo_v2_results.csv"
ML_RESULTS = BASE / "ml_scheduler_results.csv"
RANK_RESULTS = BASE / "ml_ranker_v3_scheduler_results.csv"

# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None

df = load_csv(DATASET)
ppo = load_csv(PPO_RESULTS)
ml = load_csv(ML_RESULTS)
rank = load_csv(RANK_RESULTS)

# ============================================================
# ROBUST RESULT METRIC EXTRACTION
# ============================================================
# IMPORTANT:
# The CSV files produced by the different evaluation scripts do not
# necessarily have identical column names.
#
# Never assume that PPO/ML result files contain
# "bottleneck_pressure". The functions below detect the actual
# available columns and derive a common pressure metric.

def _get_numeric_column(frame, names):
    """Return the first existing column as numeric data."""
    for name in names:
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce")
            if values.notna().any():
                return values
    return None


def _get_pressure(frame):
    """
    Find the pressure metric used by an evaluation result.

    Priority:
    1. direct selected/result pressure
    2. bottleneck pressure
    3. CPU + memory pressure -> max(CPU, memory)
    """
    direct = _get_numeric_column(
        frame,
        [
            "pressure",
            "bottleneck_pressure",
            "selected_pressure",
            "actual_pressure",
            "resource_pressure",
            "average_pressure",
            "avg_pressure",
        ],
    )

    if direct is not None:
        return direct

    cpu_pressure = _get_numeric_column(
        frame,
        [
            "cpu_pressure",
            "selected_cpu_pressure",
            "estimated_cpu_pressure",
        ],
    )

    mem_pressure = _get_numeric_column(
        frame,
        [
            "mem_pressure",
            "memory_pressure",
            "selected_mem_pressure",
            "estimated_mem_pressure",
        ],
    )

    if cpu_pressure is not None and mem_pressure is not None:
        return pd.concat(
            [cpu_pressure, mem_pressure],
            axis=1
        ).max(axis=1)

    return cpu_pressure if cpu_pressure is not None else mem_pressure


def _get_cpu_util(frame):
    return _get_numeric_column(
        frame,
        [
            "cpu_util",
            "cpu_utilization",
            "average_cpu_utilization",
            "avg_cpu_utilization",
            "selected_cpu_util",
        ],
    )


def _get_memory_util(frame):
    return _get_numeric_column(
        frame,
        [
            "mem_util",
            "memory_util",
            "memory_utilization",
            "average_memory_utilization",
            "avg_memory_utilization",
            "selected_mem_util",
        ],
    )


def _mean_or_nan(series):
    if series is None:
        return np.nan

    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) == 0:
        return np.nan

    return float(series.mean())


def evaluate_frame(frame, source_name="result"):
    """
    Convert any scheduler result CSV into the common metrics expected
    by the dashboard.

    This function NEVER directly accesses:
        frame["bottleneck_pressure"]

    Therefore a missing bottleneck_pressure column cannot cause the
    previous KeyError.
    """
    if frame is None or frame.empty:
        return None

    pressure = _get_pressure(frame)
    cpu = _get_cpu_util(frame)
    memory = _get_memory_util(frame)

    # If pressure cannot be determined, return a diagnostic object
    # instead of crashing Streamlit.
    if pressure is None:
        return {
            "valid_pressure": False,
            "source": source_name,
            "tasks": len(frame),
            "average_pressure": np.nan,
            "median_pressure": np.nan,
            "pressure_90": 0,
            "pressure_95": 0,
            "pressure_98": 0,
            "cpu": _mean_or_nan(cpu),
            "memory": _mean_or_nan(memory),
            "columns": list(frame.columns),
        }

    pressure = pd.to_numeric(
        pressure,
        errors="coerce"
    ).dropna()

    return {
        "valid_pressure": True,
        "source": source_name,
        "tasks": len(pressure),
        "average_pressure": float(pressure.mean()),
        "median_pressure": float(pressure.median()),
        "pressure_90": int((pressure >= 0.90).sum()),
        "pressure_95": int((pressure >= 0.95).sum()),
        "pressure_98": int((pressure >= 0.98).sum()),
        "cpu": _mean_or_nan(cpu),
        "memory": _mean_or_nan(memory),
        "columns": list(frame.columns),
    }


def traditional_test():
    if df is None:
        return None
    candidates = df[df["candidate_rank"] == 1].copy()
    tasks = df["instance_name"].drop_duplicates()
    # Recreate the same 70/15/15 task split used by ML/PPO evaluation.
    unique_tasks = (
        tasks.sample(frac=1.0, random_state=42).tolist()
    )
    test_tasks = set(unique_tasks[int(.85 * len(unique_tasks)):])
    return evaluate_frame(
        candidates[candidates["instance_name"].isin(test_tasks)]
    )

baseline = traditional_test()

# These files may have different schemas. The robust evaluator above
# handles them independently.
ppo_eval = evaluate_frame(ppo, "PPO v2")
ml_eval = evaluate_frame(ml, "XGBoost v2")
rank_eval = evaluate_frame(rank, "XGBoost Ranker v3")


def _show_schema_warning(name, frame, result):
    if frame is None:
        return

    if result is not None and not result.get("valid_pressure", True):
        st.warning(
            f"{name}: pressure could not be detected. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )


_show_schema_warning("PPO v2", ppo, ppo_eval)
_show_schema_warning("XGBoost v2", ml, ml_eval)
_show_schema_warning("XGBoost Ranker v3", rank, rank_eval)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## ⚙️ Dynamic CPU Scheduler")
st.sidebar.caption("Review 2 • ML + Reinforcement Learning")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "System Pipeline",
        "Scheduler Comparison",
        "Reinforcement Learning",
        "Machine Learning",
        "Dataset Explorer",
        "Task Simulator",
        "Review 2 Roadmap",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span class="tag info">Alibaba Cluster Dataset</span>'
    '<span class="tag">Gymnasium</span>'
    '<span class="tag">PPO</span>'
    '<span class="tag">XGBoost</span>',
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================
def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    hero(
        "Dynamic CPU Scheduling",
        "Machine Learning + Reinforcement Learning based scheduling dashboard"
    )

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Tasks", f"{df.instance_name.nunique():,}", "Alibaba workload instances")
        with c2:
            metric_card("Candidate assignments", f"{len(df):,}", "Five candidate machines per task")
        with c3:
            metric_card("Candidates / task", "5", "Machine choices evaluated")
        with c4:
            metric_card("Current phase", "Review 2", "Models under improvement")

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Project at a glance")
    st.markdown("""
    This system studies dynamic task-to-machine assignment using a real cluster workload.
    The current implementation contains a traditional pressure-based baseline, a PPO
    reinforcement-learning scheduler, and XGBoost-based machine-learning schedulers.
    The dashboard focuses on the complete experimental pipeline and transparent comparison.
    """)
    st.markdown(
        '<span class="tag">Data preprocessing</span>'
        '<span class="tag">Candidate generation</span>'
        '<span class="tag">RL environment</span>'
        '<span class="tag">PPO training</span>'
        '<span class="tag">ML training</span>'
        '<span class="tag">Unseen-test evaluation</span>'
        '<span class="tag">Comparison</span>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Current research position")
    st.info(
        "The current results establish the traditional scheduler as a strong baseline. "
        "PPO v2 is approximately at baseline level, while the current standalone ML "
        "models are not yet outperforming it. The next phase focuses on improving the "
        "RL reward/objective and ML objective, followed by hybrid ML + RL experiments."
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SYSTEM PIPELINE
# ============================================================
elif page == "System Pipeline":
    hero(
        "End-to-End System Pipeline",
        "From raw cluster workload to scheduling decisions and evaluation"
    )

    steps = [
        ("01", "Raw Alibaba workload", "Task execution traces and machine resource information form the source workload."),
        ("02", "Data preprocessing", "Clean records, remove duplicates, validate fields and prepare task-machine relationships."),
        ("03", "Candidate generation", "Each task is associated with five candidate machines for scheduling decisions."),
        ("04", "Feature engineering", "CPU, memory, demand, utilization, pressure, duration and machine-capacity features are prepared."),
        ("05", "Traditional baseline", "A pressure-based heuristic provides the reference scheduler for every experiment."),
        ("06", "RL environment", "Gymnasium exposes a 30-dimensional observation and five discrete machine actions."),
        ("07", "PPO training", "PPO learns a scheduling policy from repeated interactions with the environment."),
        ("08", "ML training", "XGBoost regression and learning-to-rank models learn candidate-selection relationships."),
        ("09", "Unseen evaluation", "Models are evaluated on task-level held-out data rather than training tasks."),
        ("10", "Comparison", "Pressure, utilization, candidate choices and scheduler behavior are compared."),
        ("11", "Review 2 improvement", "Reward/objective design and hybrid ML + RL scheduling are the next development targets."),
    ]

    for i in range(0, len(steps), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j >= len(steps):
                continue
            n, title, text = steps[i + j]
            with col:
                st.markdown(
                    f"""
                    <div class="step">
                        <div class="step-num">{n}</div>
                        <div class="step-title">{title}</div>
                        <div class="step-text">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Architecture")
    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Arial", fontcolor="white", color="#315675"];
        data [label="Alibaba Cluster Data", fillcolor="#153553"];
        prep [label="Preprocessing + Features", fillcolor="#132A42"];
        cand [label="5 Candidate Machines", fillcolor="#153553"];
        base [label="Traditional Baseline", fillcolor="#123D31"];
        ppo [label="PPO / RL", fillcolor="#4A3518"];
        ml [label="XGBoost / ML", fillcolor="#153553"];
        eval [label="Unseen Test Evaluation", fillcolor="#132A42"];
        ui [label="Streamlit Dashboard", fillcolor="#153553"];

        data -> prep -> cand;
        cand -> base;
        cand -> ppo;
        cand -> ml;
        base -> eval;
        ppo -> eval;
        ml -> eval;
        eval -> ui;
    }
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SCHEDULER COMPARISON
# ============================================================
elif page == "Scheduler Comparison":
    hero(
        "Scheduler Comparison",
        "Actual evaluation outputs loaded from the project result files"
    )

    records = []

    for name, result in [
        ("Traditional", baseline),
        ("PPO v2", ppo_eval),
        ("XGBoost v2", ml_eval),
        ("XGBoost Ranker v3", rank_eval),
    ]:
        if result is not None and result.get("valid_pressure", True):
            records.append((name, result))

    if records:
        comparison = pd.DataFrame([
            {
                "Scheduler": name,
                "Average pressure": result["average_pressure"],
                "Median pressure": result["median_pressure"],
                "≥90% pressure": result["pressure_90"],
                "≥95% pressure": result["pressure_95"],
                "≥98% pressure": result["pressure_98"],
                "CPU utilization": result["cpu"],
                "Memory utilization": result["memory"],
            }
            for name, result in records
        ])

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                comparison,
                x="Scheduler",
                y="Average pressure",
                title="Average resource pressure",
                template="plotly_dark",
                text_auto=".3f",
            )
            fig.update_layout(height=420, margin=dict(l=20,r=20,t=60,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                comparison,
                x="Scheduler",
                y="Median pressure",
                title="Median resource pressure",
                template="plotly_dark",
                text_auto=".2f",
            )
            fig.update_layout(height=420, margin=dict(l=20,r=20,t=60,b=20))
            st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            comparison,
            x="Scheduler",
            y=["≥90% pressure", "≥95% pressure", "≥98% pressure"],
            title="High-pressure task counts",
            barmode="group",
            template="plotly_dark",
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                comparison,
                x="Scheduler",
                y="CPU utilization",
                title="Average CPU utilization",
                template="plotly_dark",
                text_auto=".2f",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(
                comparison,
                x="Scheduler",
                y="Memory utilization",
                title="Average memory utilization",
                template="plotly_dark",
                text_auto=".2f",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Evaluation table")
        st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)

# ============================================================
# RL
# ============================================================
elif page == "Reinforcement Learning":
    hero(
        "Reinforcement Learning",
        "PPO-based dynamic scheduling using a Gymnasium environment"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Observation", "30 features", "Normalized task + machine state")
    with c2:
        metric_card("Action space", "5 actions", "Select one candidate machine")
    with c3:
        metric_card("Training", "501,760", "PPO v2 timesteps")

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("RL decision loop")
    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Arial", fontcolor="white", color="#315675"];
        obs [label="30-D Observation", fillcolor="#153553"];
        ppo [label="PPO Policy", fillcolor="#4A3518"];
        act [label="Select 1 of 5", fillcolor="#153553"];
        reward [label="Reward", fillcolor="#123D31"];
        obs -> ppo -> act -> reward -> obs;
    }
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    if ppo is not None:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("PPO v2 selected candidates")
        counts = (
            ppo["candidate_rank"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        counts.columns = ["Candidate", "Selections"]
        fig = px.bar(
            counts,
            x="Candidate",
            y="Selections",
            title="PPO v2 candidate selection distribution",
            template="plotly_dark",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("RL development status")
    st.success("Environment validation completed.")
    st.success("PPO v1 training completed.")
    st.success("PPO v2 training and policy recovery completed.")
    st.warning("PPO v2 currently reproduces the traditional baseline rather than exceeding it.")
    st.info("Reward design improvement is planned for the next development phase.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ML
# ============================================================
elif page == "Machine Learning":
    hero(
        "Machine Learning",
        "XGBoost regression and learning-to-rank experiments"
    )

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("ML development progression")

    stages = pd.DataFrame({
        "Stage": ["XGBoost v1", "XGBoost v2", "XGBoost Ranker v3"],
        "Purpose": [
            "Assignment-cost regression",
            "Improved cost prediction",
            "Candidate ranking"
        ],
        "Status": [
            "Target-derived feature issue identified",
            "Trained and evaluated",
            "Trained and evaluated"
        ]
    })
    st.dataframe(stages, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if rank is not None:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("ML Ranker v3 candidate behavior")
        counts = (
            rank["candidate_rank"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        counts.columns = ["Candidate", "Selections"]
        fig = px.bar(
            counts,
            x="Candidate",
            y="Selections",
            title="Ranker v3 selected candidate distribution",
            template="plotly_dark",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("ML workflow")
    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Arial", fontcolor="white", color="#315675"];
        raw [label="44,895 rows", fillcolor="#153553"];
        clean [label="Clean + validate", fillcolor="#132A42"];
        split [label="Task-level 70/15/15 split", fillcolor="#153553"];
        train [label="XGBoost training", fillcolor="#4A3518"];
        test [label="Unseen test", fillcolor="#123D31"];
        compare [label="Scheduler comparison", fillcolor="#153553"];
        raw -> clean -> split -> train -> test -> compare;
    }
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    hero(
        "Dataset Explorer",
        "Interactive view of task-machine candidates"
    )

    if df is None:
        st.error("task_machine_candidates_v2.csv was not found.")
    else:
        work = df.copy()

        col1, col2, col3 = st.columns(3)

        with col1:
            ranks = st.multiselect(
                "Candidate rank",
                sorted(work["candidate_rank"].unique()),
                default=sorted(work["candidate_rank"].unique()),
            )

        with col2:
            cpu_limit = st.slider(
                "Maximum CPU pressure",
                0.0, 1.0, 1.0, 0.01
            )

        with col3:
            mem_limit = st.slider(
                "Maximum memory pressure",
                0.0, 1.0, 1.0, 0.01
            )

        work = work[
            work["candidate_rank"].isin(ranks)
            & (work["cpu_pressure"] <= cpu_limit)
            & (work["mem_pressure"] <= mem_limit)
        ]

        st.write(f"Showing {len(work):,} candidate rows")

        st.dataframe(
            work[
                [
                    "event_time",
                    "instance_name",
                    "machine_id",
                    "candidate_rank",
                    "cpu_util",
                    "mem_util",
                    "cpu_pressure",
                    "mem_pressure",
                    "bottleneck_pressure",
                    "task_duration",
                    "cpu_demand_ratio",
                    "mem_demand_ratio",
                ]
            ].head(500),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# TASK SIMULATOR
# ============================================================
elif page == "Task Simulator":
    hero(
        "Task Simulator",
        "Inspect the five candidate machines for an individual workload"
    )

    if df is None:
        st.error("Dataset not found.")
    else:
        tasks = sorted(df["instance_name"].unique())

        selected = st.selectbox(
            "Select task",
            tasks
        )

        task = df[
            df["instance_name"] == selected
        ].copy()

        st.markdown('<div class="section">', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        row = task.iloc[0]

        with c1:
            metric_card("Task", selected, str(row["task_name"]))
        with c2:
            metric_card("Task type", str(row["task_type"]), "Workload category")
        with c3:
            metric_card("Duration", f'{row["task_duration"]:.2f}', "Observed duration")
        with c4:
            metric_card("Candidates", "5", "Available machine choices")

        st.markdown('</div>', unsafe_allow_html=True)

        display = task[
            [
                "candidate_rank",
                "machine_id",
                "cpu_num",
                "mem_size",
                "cpu_util",
                "mem_util",
                "cpu_pressure",
                "mem_pressure",
                "bottleneck_pressure",
                "estimated_cpu_pressure",
                "estimated_mem_pressure",
            ]
        ].sort_values("candidate_rank")

        st.subheader("Candidate machines")
        st.dataframe(
            display.style.format({
                "cpu_util": "{:.3f}",
                "mem_util": "{:.3f}",
                "cpu_pressure": "{:.3f}",
                "mem_pressure": "{:.3f}",
                "bottleneck_pressure": "{:.3f}",
                "estimated_cpu_pressure": "{:.3f}",
                "estimated_mem_pressure": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            display,
            x="candidate_rank",
            y=["cpu_pressure", "mem_pressure", "bottleneck_pressure"],
            title="Candidate resource pressure",
            barmode="group",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ROADMAP
# ============================================================
elif page == "Review 2 Roadmap":
    hero(
        "Review 2 — Current Status & Next Phase",
        "A transparent view of what has been completed and what will be improved"
    )

    completed = [
        "Alibaba cluster workload preparation",
        "Task-machine candidate generation",
        "Feature engineering",
        "Traditional scheduling baseline",
        "Gymnasium environment",
        "PPO v1",
        "PPO v2",
        "XGBoost v1",
        "XGBoost v2",
        "XGBoost learning-to-rank v3",
        "Unseen test evaluation",
        "Scheduler comparison",
        "Interactive dashboard",
    ]

    for item in completed:
        st.markdown(
            f'<span class="tag good">✓ {item}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Current findings")
    st.markdown("""
    **Traditional baseline:** strongest standalone scheduler in the current experiment.

    **PPO v2:** approximately reproduces the baseline, showing that the RL pipeline is
    functioning but the current reward design does not yet produce an improvement.

    **XGBoost:** demonstrates strong predictive/ranking capability, but the current ML
    objectives have not yet translated into lower actual scheduling pressure.

    **Research direction:** improve the reward/objective functions and investigate a
    hybrid ML + RL scheduling architecture.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Next development cycle")
    next_steps = [
        ("01", "Improve RL reward", "Better balance pressure, resource utilization, duration and workload distribution."),
        ("02", "Improve ML objective", "Move from indirect cost prediction toward a scheduling objective aligned with evaluation."),
        ("03", "Hybrid ML + RL", "Use ML for resource/performance prediction and RL for adaptive scheduling decisions."),
        ("04", "Expanded evaluation", "Compare pressure, utilization, workload distribution and scheduling stability."),
        ("05", "Final model selection", "Select the best standalone or hybrid strategy for the final demonstration."),
    ]

    for i in range(0, len(next_steps), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j >= len(next_steps):
                continue
            n, title, text = next_steps[i+j]
            with col:
                st.markdown(
                    f"""
                    <div class="step">
                        <div class="step-num">{n}</div>
                        <div class="step-title">{title}</div>
                        <div class="step-text">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer">Dynamic CPU Scheduling • Review 2 • '
    'Machine Learning + Reinforcement Learning</div>',
    unsafe_allow_html=True,
)
