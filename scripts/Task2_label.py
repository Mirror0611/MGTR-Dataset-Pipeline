import re
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import json
from tqdm import tqdm

# ================= 配置区 =================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MIMIC = r"D:\科研\电子病历关系与时间建模\datasets\MIMIC-III-clinical-database\mimic-iii-clinical-database-1.4"
BASE_PATH = os.environ.get("MIMIC_III_PATH", _DEFAULT_MIMIC)
KB_FILE = os.path.join(_SCRIPT_DIR, "knowledge_base.json")
SAMPLE_LIMIT = 58976
# SAMPLE_LIMIT = 5000
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "Task2_all.csv")
MIN_OBSERVATIONS = 5
OBSERVATION_WINDOW_DAYS = 14
RANDOM_SEED = 42
# INPUT_TEXT 使用的观测比例：1.0 = 窗口内全部检验点（与打标序列一致，便于模型学习靶区/毒性模式）；
# 0.4 等 <1 表示仅早期部分，适合「仅用前期化验预测整窗结局」的前瞻设定（更难）。
EARLY_OBS_FRACTION = 1.0


def load_knowledge_base(filepath):
    """加载外部医学知识库"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到知识库文件: {filepath}。请确保已创建该JSON文件。")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ================= 核心算法：基于指南规则的临床判定 =================
def evaluate_clinical_trajectory(vals, rule_config):
    """
    基于临床指南设定的靶区和毒性阈值进行 4 分类打标。
    优先级：1. TOXICITY -> 2. REBOUND -> 3. TARGET_ACHIEVED -> 4. REFRACTORY
    """
    if len(vals) < MIN_OBSERVATIONS:
        return "UNKNOWN"

    v_start = vals[0]
    v_end = vals[-1]
    v_max = max(vals)
    v_min = min(vals)

    rule_type = rule_config.get("rule_type", "absolute")

    if rule_type == "absolute":
        t_min = rule_config["target_min"]
        t_max = rule_config["target_max"]
        tox_min = rule_config["tox_min"]
        tox_max = rule_config["tox_max"]

        if v_min < tox_min or v_max > tox_max:
            return "TOXICITY"

        ever_in_target = any(t_min <= v <= t_max for v in vals)
        end_in_target = t_min <= v_end <= t_max

        if ever_in_target and not end_in_target:
            return "REBOUND"

        if end_in_target:
            return "TARGET_ACHIEVED"

        return "REFRACTORY"

    elif rule_type == "relative_baseline":
        tox_abs_inc = rule_config.get("tox_abs_increase", 999)
        tox_pct_inc = rule_config.get("tox_pct_increase", 999)

        max_increase = v_max - v_start
        max_pct_increase = (v_max - v_start) / v_start if v_start > 0 else 0.0
        end_increase = v_end - v_start
        end_pct_increase = (v_end - v_start) / v_start if v_start > 0 else 0.0

        if max_increase >= tox_abs_inc or max_pct_increase >= tox_pct_inc:
            return "TOXICITY"

        pct_primary = rule_config.get("relative_pct_primary", False) or tox_abs_inc >= 900

        if pct_primary:
            if tox_pct_inc >= 900:
                return "REFRACTORY"
            half = tox_pct_inc * 0.5
            if end_pct_increase <= half and max_pct_increase < tox_pct_inc:
                return "TARGET_ACHIEVED"
            return "REFRACTORY"

        if end_increase < (tox_abs_inc * 0.5):
            return "TARGET_ACHIEVED"

        return "REFRACTORY"

    return "UNKNOWN"


def _build_input_text(d_name, seq_labs_early, t_start, obs_window_days, early_obs_count, total_obs_count):
    """根据是否使用全窗观测生成英文任务描述。"""
    actual_early_days = (seq_labs_early["CHARTTIME"].iloc[-1] - t_start).total_seconds() / 86400
    timeline_texts = []
    for _, row in seq_labs_early.iterrows():
        offset_days = (row["CHARTTIME"] - t_start).total_seconds() / 86400
        val = row["VALUENUM"]
        timeline_texts.append(f"Day {offset_days:.1f}: {val:.1f}")
    timeline_str = ", ".join(timeline_texts)

    if early_obs_count >= total_obs_count:
        return (
            f"Patient was administered {d_name}. "
            f"Based on all {early_obs_count} laboratory observations spanning {actual_early_days:.1f} days "
            f"within the {obs_window_days}-day monitoring window, measurement timeline: {timeline_str}. "
            f"Classify the guideline-based trajectory pattern for this period."
        )

    return (
        f"Patient was administered {d_name}. "
        f"Based on the first {early_obs_count} early observations spanning {actual_early_days:.1f} days "
        f"(out of {total_obs_count} measurements in the {obs_window_days}-day window), measurement timeline: {timeline_str}. "
        f"Predict the overall mid-term (up to {obs_window_days} days) trajectory pattern."
    )


# ================= 主构建流程 =================
def build_task2_dataset():
    rng = np.random.RandomState(RANDOM_SEED)

    evidence_map = load_knowledge_base(KB_FILE)
    print(f"已加载外部医学规则库，包含 {len(evidence_map)} 种药物配置。")
    print(f"MIMIC 数据目录: {BASE_PATH}")
    print(f"EARLY_OBS_FRACTION={EARLY_OBS_FRACTION}（1.0 表示 INPUT_TEXT 与打标使用同一套时间序列）")

    print("Step 1: 正在加载元数据与药物记录...")
    d_lab = pd.read_csv(os.path.join(BASE_PATH, "D_LABITEMS.csv"))
    itemid_to_label = d_lab.set_index("ITEMID")["LABEL"].to_dict()

    pres = pd.read_csv(os.path.join(BASE_PATH, "PRESCRIPTIONS.csv"), low_memory=False)
    pres["STARTDATE"] = pd.to_datetime(pres["STARTDATE"])

    drug_patterns = [re.escape(k) for k in evidence_map.keys()]
    pattern = "|".join(drug_patterns)
    target_drugs = pres[pres["DRUG"].str.contains(pattern, case=False, na=False, regex=True)].copy()

    unique_all = target_drugs["HADM_ID"].unique()
    if len(unique_all) <= SAMPLE_LIMIT:
        unique_hadms = unique_all
    else:
        unique_hadms = rng.choice(unique_all, size=SAMPLE_LIMIT, replace=False)
    unique_hadms = np.sort(unique_hadms)
    unique_hadms_set = set(unique_hadms)

    target_drugs = target_drugs[target_drugs["HADM_ID"].isin(unique_hadms_set)]
    drugs_by_hadm = target_drugs.groupby("HADM_ID")

    sorted_drug_keys = sorted(evidence_map.keys(), key=len, reverse=True)

    print(f"Step 2: 正在加载实验室指标 (涉及 {len(unique_hadms)} 个住院号)...")
    relevant_itemids = set(iid for meta in evidence_map.values() for iid in meta["itemids"])
    labs_list = []
    for chunk in pd.read_csv(
        os.path.join(BASE_PATH, "LABEVENTS.csv"), chunksize=1000000, low_memory=False
    ):
        filtered = chunk[chunk["HADM_ID"].isin(unique_hadms_set) & chunk["ITEMID"].isin(relevant_itemids)]
        labs_list.append(filtered)
    labs = pd.concat(labs_list)
    labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"])
    labs = labs.dropna(subset=["VALUENUM"])

    dataset = []

    print("Step 3: 执行知识驱动的时序打标...")
    labs_grouped = labs.groupby("HADM_ID")

    for hadm_id in tqdm(unique_hadms):
        if hadm_id not in labs_grouped.groups:
            continue
        if hadm_id not in drugs_by_hadm.groups:
            continue
        p_drugs = drugs_by_hadm.get_group(hadm_id)
        p_labs = labs_grouped.get_group(hadm_id)

        first_admins = p_drugs.groupby("DRUG")["STARTDATE"].min().reset_index()

        for _, drug_row in first_admins.iterrows():
            d_name = drug_row["DRUG"]
            t_start = drug_row["STARTDATE"]
            t_end = t_start + timedelta(days=OBSERVATION_WINDOW_DAYS)

            current_config = next(
                (evidence_map[k] for k in sorted_drug_keys if k.lower() in d_name.lower()),
                None,
            )
            if not current_config:
                continue

            for iid in current_config["itemids"]:
                seq_labs = p_labs[
                    (p_labs["ITEMID"] == iid)
                    & (p_labs["CHARTTIME"] >= t_start)
                    & (p_labs["CHARTTIME"] <= t_end)
                ].copy()

                valid_min, valid_max = current_config["valid_range"]
                seq_labs = seq_labs[
                    (seq_labs["VALUENUM"] >= valid_min) & (seq_labs["VALUENUM"] <= valid_max)
                ]

                seq_labs = seq_labs.sort_values(by="CHARTTIME").drop_duplicates(
                    subset=["CHARTTIME"], keep="last"
                )

                if len(seq_labs) < MIN_OBSERVATIONS:
                    continue

                lab_label = itemid_to_label.get(iid, current_config["name"])
                vals = seq_labs["VALUENUM"].tolist()
                obs_count = len(vals)
                actual_days = (seq_labs["CHARTTIME"].iloc[-1] - t_start).total_seconds() / 86400

                pattern_label = evaluate_clinical_trajectory(vals, current_config)

                if pattern_label == "UNKNOWN":
                    continue

                if EARLY_OBS_FRACTION >= 1.0:
                    early_obs_count = obs_count
                    early_labs = seq_labs
                else:
                    early_obs_count = max(2, int(obs_count * EARLY_OBS_FRACTION))
                    early_obs_count = min(early_obs_count, obs_count - 1)
                    early_labs = seq_labs.head(early_obs_count)

                input_text = _build_input_text(
                    d_name,
                    early_labs,
                    t_start,
                    OBSERVATION_WINDOW_DAYS,
                    early_obs_count,
                    obs_count,
                )

                dataset.append(
                    {
                        "HADM_ID": hadm_id,
                        "DRUG": d_name,
                        "LAB_NAME": lab_label,
                        "OBS_COUNT": obs_count,
                        "WINDOW_DAYS": round(actual_days, 2),
                        "LABEL": pattern_label,
                        "INPUT_TEXT": input_text,
                    }
                )

    df_res = pd.DataFrame(dataset)
    if not df_res.empty:
        df_res.to_csv(OUTPUT_FILE, index=False)
        print("\nTask-2 数据处理与规则清洗完成！")
        print(f"总序列样本数: {len(df_res)}")
        print("\n医学标签分布情况：")
        print(df_res["LABEL"].value_counts())
    else:
        print("未提取到满足条件的序列样本。")


if __name__ == "__main__":
    build_task2_dataset()
